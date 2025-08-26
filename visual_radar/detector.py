from __future__ import annotations

from collections import deque
from typing import List, Set, Tuple, Optional

import numpy as np
import cv2 as cv

from visual_radar.config import SMDParams
from visual_radar.motion import (
    as_gray,
    DualBGModel,
    find_motion_bboxes,
    make_masks_static_and_slow,
)
from visual_radar.stereo import gate_pairs_rectified, epipolar_ncc_match
from visual_radar.utils import BBox


# --- helpers ------------------------------------------------------------------


def _vr_pre(gray: np.ndarray, night: bool) -> np.ndarray:
    """
    Light denoising at night (supports even/odd kernels as per OpenCV docs).
    """
    if night:
        try:
            gray = cv.fastNlMeansDenoising(gray, None, 7, 7, 21)
        except Exception:
            pass
        gray = cv.GaussianBlur(gray, (5, 5), 0)
    return gray


def _boxes_from_mask(mask: np.ndarray, min_area: int, max_area: int) -> List[BBox]:
    """
    Extract bounding boxes from a binary mask using contours.
    """
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
        boxes.append(BBox(float(x), float(y), float(w), float(h)))
    return boxes


def _shape_gate_sky(
    boxes: List[BBox],
    h: int,
    split: float,
    max_ar_w: float = 6.0,
    max_sky_w: int = 240,
    min_h: int = 3,
) -> List[BBox]:
    """
    Shape filter for sky region: removes long horizontal cloud stripes.
    """
    out: List[BBox] = []
    thr_y = split * float(h)
    for b in boxes:
        if b.cy < thr_y:
            arw = b.w / max(1.0, b.h)
            if arw > max_ar_w or b.w > max_sky_w or b.h < float(min_h):
                continue
        out.append(b)
    return out


def _sails_only_gate(
    boxes: List[BBox],
    gray: np.ndarray,
    water_y: int,
    white_delta: float = 18.0,
    min_h_over_w: float = 0.85,
    max_w: int = 240,
    min_h: int = 6,
) -> List[BBox]:
    """
    Keep only candidate sail boxes:
    - Centroid below water_y (water region)
    - Tall box: h/w >= min_h_over_w, h >= min_h, w <= max_w
    - Brighter than water: mean(gray_box) > median(gray_water_band) + white_delta
    """
    h, w = gray.shape[:2]
    y1 = max(water_y - 40, 0)
    y2 = min(water_y + 80, h)
    water_band = gray[y1:y2, :]
    base_med = float(np.median(water_band)) if water_band.size else float(np.median(gray))

    out: List[BBox] = []
    for b in boxes:
        if b.cy < float(water_y):
            continue
        if (b.h / max(1.0, b.w)) < float(min_h_over_w):
            continue
        if b.h < float(min_h) or b.w > float(max_w):
            continue
        x1, y1, x2, y2 = int(b.x), int(b.y), int(b.x + b.w), int(b.y + b.h)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        med_box = float(np.mean(gray[y1:y2, x1:x2]))
        if med_box > (base_med + float(white_delta)):
            out.append(b)
    return out


# --- detector -----------------------------------------------------------------


class StereoMotionDetector:
    """
    Main detector for a pair of *rectified* frames.
    Produces (maskL, maskR), (boxesL, boxesR) and stereo pairs.
    """

    def __init__(self, frame_size: Tuple[int, int], params: SMDParams):
        self._frame_size = frame_size  # (w, h)
        self.params = params

        w, h = frame_size
        self.bgL = DualBGModel((h, w))
        self.bgR = DualBGModel((h, w))

        self.persist_k = int(getattr(self.params, "persist_k", 4))
        self.persist_m = int(getattr(self.params, "persist_m", 3))
        self.histL: deque[np.ndarray] = deque(maxlen=self.persist_k)
        self.histR: deque[np.ndarray] = deque(maxlen=self.persist_k)

        self.is_night: bool = False
        self._drift_streak: int = 0

        # Optional ROI mask
        self.roi: Optional[np.ndarray] = None
        roi_path = getattr(self.params, "roi_mask", None)
        if roi_path:
            try:
                m = cv.imread(str(roi_path), cv.IMREAD_GRAYSCALE)
                if m is not None:
                    W, H = frame_size
                    if m.shape[:2] != (H, W):
                        m = cv.resize(m, (W, H), interpolation=cv.INTER_NEAREST)
                    self.roi = (m > 0).astype(np.uint8) * 255
            except Exception:
                self.roi = None

    # internal: reset background models and persistence
    def _reset_background(self):
        w, h = self._frame_size
        try:
            self.bgL = DualBGModel((h, w))
            self.bgR = DualBGModel((h, w))
        except Exception:
            pass
        self.histL = deque(maxlen=self.persist_k)
        self.histR = deque(maxlen=self.persist_k)

    # main step
    def step(self, rectL_bgr: np.ndarray, rectR_bgr: np.ndarray):
        """
        Process a pair of rectified BGR frames and return masks, boxes, and stereo pairs.
        """
        gL = as_gray(rectL_bgr)
        gR = as_gray(rectR_bgr)

        # Night/day
        night_auto = bool(getattr(self.params, "night_auto", True))
        night_thr = float(getattr(self.params, "night_luma_thr", 50.0))
        self.is_night = (float(np.mean(gL)) < night_thr) if night_auto else False

        gL = _vr_pre(gL, self.is_night)
        gR = _vr_pre(gR, self.is_night)

        use_clahe = bool(getattr(self.params, "use_clahe", True))
        if self.is_night:
            # CLAHE at night increases false positives; disable
            use_clahe = False

        # --- base motion masks (fast + morphological cleanup inside)
        mL, _ = find_motion_bboxes(
            gL, self.bgL,
            int(getattr(self.params, "min_area", 25)),
            int(getattr(self.params, "max_area", 0)),
            float(getattr(self.params, "thr_fast", 2.0)),
            float(getattr(self.params, "thr_slow", 1.0)),
            use_clahe=use_clahe,
            size_aware_morph=bool(getattr(self.params, "size_aware_morph", True)),
        )
        mR, _ = find_motion_bboxes(
            gR, self.bgR,
            int(getattr(self.params, "min_area", 25)),
            int(getattr(self.params, "max_area", 0)),
            float(getattr(self.params, "thr_fast", 2.0)),
            float(getattr(self.params, "thr_slow", 1.0)),
            use_clahe=use_clahe,
            size_aware_morph=bool(getattr(self.params, "size_aware_morph", True)),
        )

        # --- SKY: keep only FAST motion (birds), drop slow cloud drift
        try:
            mL_static, mL_slow = make_masks_static_and_slow(
                gL, self.bgL,
                float(getattr(self.params, "thr_fast", 2.0)),
                float(getattr(self.params, "thr_slow", 1.0)),
                use_clahe=use_clahe, kernel_size=3,
            )
            mR_static, mR_slow = make_masks_static_and_slow(
                gR, self.bgR,
                float(getattr(self.params, "thr_fast", 2.0)),
                float(getattr(self.params, "thr_slow", 1.0)),
                use_clahe=use_clahe, kernel_size=3,
            )

            h2, w2 = gL.shape[:2]
            split = float(getattr(self.params, "y_area_split", 0.55))
            sky_y = int(split * h2)

            # fast = base - slow  (остаются быстрые пиксели)
            mL_fast = cv.bitwise_and(mL, cv.bitwise_not(mL_slow))
            mR_fast = cv.bitwise_and(mR, cv.bitwise_not(mR_slow))

            k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            mL_fast = cv.morphologyEx(mL_fast, cv.MORPH_OPEN, k3, iterations=1)
            mR_fast = cv.morphologyEx(mR_fast, cv.MORPH_OPEN, k3, iterations=1)

            # Применяем только к верхней части (небо)
            mL[:sky_y, :] = mL_fast[:sky_y, :]
            mR[:sky_y, :] = mR_fast[:sky_y, :]
        except Exception:
            pass

        # --- optional crop from top
        crop_top = int(getattr(self.params, "crop_top", 0))
        if crop_top > 0:
            mL[:crop_top, :] = 0
            mR[:crop_top, :] = 0

        # --- ROI mask
        if self.roi is not None:
            h_cur, w_cur = mL.shape[:2]
            if self.roi.shape[:2] != (h_cur, w_cur):
                roi = cv.resize(self.roi, (w_cur, h_cur), interpolation=cv.INTER_NEAREST)
                roi = (roi > 0).astype(np.uint8) * 255
            else:
                roi = self.roi
            mL = cv.bitwise_and(mL, roi)
            mR = cv.bitwise_and(mR, roi)

        # --- adjust min_area at night
        min_area = int(getattr(self.params, "min_area", 25))
        if self.is_night:
            mult = float(getattr(self.params, "min_area_night_mult", 4.0))
            min_area = int(min_area * mult)
        max_area = int(getattr(self.params, "max_area", 0))

        # --- CC cleanup then persistence
        def _cc_clean(mm: np.ndarray) -> np.ndarray:
            if mm is None or mm.size == 0:
                return mm
            num, labels, stats, _ = cv.connectedComponentsWithStats(mm, 8)
            cleaned = np.zeros_like(mm)
            for i in range(1, num):
                a = stats[i, cv.CC_STAT_AREA]
                if a >= min_area and (max_area == 0 or a <= max_area):
                    cleaned[labels == i] = 255
            return cleaned

        mL_clean = _cc_clean(mL)
        mR_clean = _cc_clean(mR)

        def _persist_mask(hist, cur, need):
            S = np.zeros_like(cur, dtype=np.uint16)
            for x in hist:
                if x is not None:
                    S += x.astype(np.uint16)
            S += cur.astype(np.uint16)
            thr = 255 * int(need)
            return (S >= thr).astype(np.uint8) * 255

        self.histL.append(mL_clean)
        self.histR.append(mR_clean)
        mL_final = _persist_mask(self.histL, mL_clean, int(getattr(self.params, "persist_m", 3)))
        mR_final = _persist_mask(self.histR, mR_clean, int(getattr(self.params, "persist_m", 3)))

        # --- anti-drift: reset BG if too much foreground for too long
        h, w = gL.shape[:2]
        fg_ratio_L = float(np.count_nonzero(mL_final)) / float(h * w)
        fg_ratio_R = float(np.count_nonzero(mR_final)) / float(h * w)
        fg_ratio = max(fg_ratio_L, fg_ratio_R)
        drift_max_fg = float(getattr(self.params, "drift_max_fg_pct", 0.25))
        drift_max_len = int(getattr(self.params, "drift_max_frames", 60))
        if fg_ratio > drift_max_fg:
            self._drift_streak += 1
        else:
            self._drift_streak = 0
        if self._drift_streak >= drift_max_len:
            self._reset_background()
            self._drift_streak = 0

        # --- masks -> boxes
        boxesL = _boxes_from_mask(mL_final, min_area=min_area, max_area=max_area)
        boxesR = _boxes_from_mask(mR_final, min_area=min_area, max_area=max_area)

        # Sky: remove very wide cloud strips in the top part
        split = float(getattr(self.params, "y_area_split", 0.55))
        boxesL = _shape_gate_sky(boxesL, h, split)
        boxesR = _shape_gate_sky(boxesR, h, split)

        # Optional "sails only" mode
        sails_only = bool(getattr(self.params, "sails_only", False))
        if self.roi is not None or sails_only:
            water_split = float(getattr(self.params, "sails_water_split", 0.55))
            water_y = int(water_split * h)
            white_delta = float(getattr(self.params, "sails_white_delta", 18.0))
            min_h_over_w = float(getattr(self.params, "sails_min_h_over_w", 0.85))
            max_w = int(getattr(self.params, "sails_max_w", 240))
            min_h = int(getattr(self.params, "sails_min_h", 6))
            boxesL = _sails_only_gate(boxesL, gL, water_y, white_delta, min_h_over_w, max_w, min_h)
            boxesR = _sails_only_gate(boxesR, gR, water_y, white_delta, min_h_over_w, max_w, min_h)

        # --- pairing and NCC refinement
        pairs = gate_pairs_rectified(
            boxesL, boxesR,
            float(getattr(self.params, "y_eps", 6.0)),
            float(getattr(self.params, "dmin", -512.0)),
            float(getattr(self.params, "dmax", 512.0)),
        )

        matched_R: Set[int] = set(j for _, j in pairs)
        for i, bl in enumerate(boxesL):
            if any(i == ii for ii, _ in pairs):
                continue
            rb = epipolar_ncc_match(
                gL, gR, bl,
                int(getattr(self.params, "stereo_search_pad", 64)),
                int(getattr(self.params, "stereo_patch", 13)),
                float(getattr(self.params, "stereo_ncc_min", 0.25)),
            )
            if rb is not None:
                boxesR.append(rb)
                j = len(boxesR) - 1
                pairs.append((i, j))
                matched_R.add(j)

        return mL_final, mR_final, boxesL, boxesR, pairs
