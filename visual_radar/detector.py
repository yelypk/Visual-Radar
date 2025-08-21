import numpy as np
import cv2 as cv
from collections import deque

from typing import Tuple, List, Set
from .config import SMDParams
from .motion import as_gray, DualBGModel, find_motion_bboxes
from .stereo import gate_pairs_rectified, epipolar_ncc_match
from .utils import BBox

def _vr_sigma(img):
    med = np.median(img)
    mad = np.median(np.abs(img - med))
    return 1.4826 * mad

def _vr_pre(gray, night: bool):
    """Сильнее приглушаем ночной шум (NLM + Gaussian)."""
    if night:
        try:
            gray = cv.fastNlMeansDenoising(gray, None, 7, 7, 21)
            gray = cv.GaussianBlur(gray, (5, 5), 0)
        except Exception:
            gray = cv.GaussianBlur(gray, (5, 5), 0)
    return gray

def _boxes_from_mask(mask: np.ndarray, min_area: int, max_area: int) -> List[BBox]:
    """BBox(x,y,w,h, area, cx, cy) из бинарной маски по контурам."""
    boxes: List[BBox] = []
    if mask is None or mask.size == 0:
        return boxes
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        area = int(w) * int(h)
        if area < max(min_area, 1):
            continue
        if max_area and area > max_area:
            continue
        cx = float(x) + float(w) / 2.0
        cy = float(y) + float(h) / 2.0
        boxes.append(BBox(float(x), float(y), float(w), float(h), float(area), cx, cy))
    return boxes

class StereoMotionDetector:
    def __init__(self, frame_size: Tuple[int,int], params: SMDParams):
        self.params = params
        w, h = frame_size
        self.bgL = DualBGModel((h, w))
        self.bgR = DualBGModel((h, w))
        # temporal persistence
        self.persist_k = int(getattr(self.params, 'persist_k', 4))  # окно
        self.persist_m = int(getattr(self.params, 'persist_m', 3))  # минимум попаданий
        self.histL = deque(maxlen=self.persist_k)
        self.histR = deque(maxlen=self.persist_k)

    def step(self, rectL_bgr, rectR_bgr):
        gL = as_gray(rectL_bgr)
        gR = as_gray(rectR_bgr)

        night_auto = bool(getattr(self.params, "night_auto", True))
        night_thr  = float(getattr(self.params, "night_luma_thr", 50.0))
        is_night   = (float(np.mean(gL)) < night_thr) if night_auto else False

        gL = _vr_pre(gL, is_night)
        gR = _vr_pre(gR, is_night)

        use_clahe = bool(getattr(self.params, 'use_clahe', True))
        if is_night:
            use_clahe = False

        mL, boxesL = find_motion_bboxes(
            gL, self.bgL,
            self.params.min_area, self.params.max_area,
            self.params.thr_fast, self.params.thr_slow,
            use_clahe=use_clahe,
            size_aware_morph=self.params.size_aware_morph
        )
        mR, boxesR = find_motion_bboxes(
            gR, self.bgR,
            self.params.min_area, self.params.max_area,
            self.params.thr_fast, self.params.thr_slow,
            use_clahe=use_clahe,
            size_aware_morph=self.params.size_aware_morph
        )

        min_area = int(getattr(self.params, "min_area", 25))
        if is_night:
            min_area = int(min_area * float(getattr(self.params, "min_area_night_mult", 4.0)))
        max_area = int(getattr(self.params, "max_area", 0))

        def _cc_clean(mm: np.ndarray) -> np.ndarray:
            if mm is None or mm.size == 0:
                return mm
            num, labels, stats, _ = cv.connectedComponentsWithStats(mm, 8)
            cleaned = np.zeros_like(mm)
            for i in range(1, num):  # 0 — фон
                a = stats[i, cv.CC_STAT_AREA]
                if a >= min_area and (max_area == 0 or a <= max_area):
                    cleaned[labels == i] = 255
            return cleaned

        mL_clean = _cc_clean(mL)
        mR_clean = _cc_clean(mR)

        def _persist_mask(hist, cur, need, K):
            S = np.zeros_like(cur, dtype=np.uint16)
            for x in hist:
                if x is not None:
                    S += x.astype(np.uint16)
            S += cur.astype(np.uint16)
            thr = 255 * int(need)  
            return (S >= thr).astype(np.uint8) * 255

        self.histL.append(mL_clean)
        self.histR.append(mR_clean)
        mL_final = _persist_mask(self.histL, mL_clean, self.persist_m, self.persist_k)
        mR_final = _persist_mask(self.histR, mR_clean, self.persist_m, self.persist_k)

        boxesL = _boxes_from_mask(mL_final, min_area=min_area, max_area=max_area)
        boxesR = _boxes_from_mask(mR_final, min_area=min_area, max_area=max_area)

        pairs = gate_pairs_rectified(boxesL, boxesR, self.params.y_eps, self.params.dmin, self.params.dmax)

        matched_R: Set[int] = set(j for _, j in pairs)
        for i, bl in enumerate(boxesL):
            if any(i == ii for ii, _ in pairs):
                continue
            rb = epipolar_ncc_match(gL, gR, bl,
                                    self.params.stereo_search_pad,
                                    self.params.stereo_patch,
                                    self.params.stereo_ncc_min)
            if rb is not None:
                boxesR.append(rb)
                j = len(boxesR) - 1
                pairs.append((i, j))
                matched_R.add(j)

        return mL_final, mR_final, boxesL, boxesR, pairs