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


def _vr_pre(gray: np.ndarray, night: bool) -> np.ndarray:
    """Лёгкое шумоподавление ночью (поддерживаем чётные/нечётные ядра согласно докам)."""
    if night:
        # non-local means затем мягкий Gaussian
        try:
            gray = cv.fastNlMeansDenoising(gray, None, 7, 7, 21)
        except Exception:
            pass
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
    """Фильтр «формы» только для НЕБА: отбрасываем длинные горизонтальные ленты облаков."""
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
    Оставляем кандидатов на ПАРУСА:
    - центроид НИЖЕ water_y (вода),
    - «высокий» бокс: h/w >= min_h_over_w, h >= min_h, w <= max_w,
    - белее воды: mean(gray_box) > median(gray_water_band) + white_delta.
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
        if b.h < min_h or b.w > max_w:
            continue
        if (b.h / max(1.0, b.w)) < min_h_over_w:
            continue

        x1 = int(max(0, b.x))
        y1b = int(max(0, b.y))
        x2 = int(min(w, b.x + b.w))
        y2b = int(min(h, b.y + b.h))
        patch = gray[y1b:y2b, x1:x2]
        if patch.size == 0:
            continue
        if float(np.mean(patch)) < base_med + white_delta:
            continue

        out.append(b)
    return out


class StereoMotionDetector:
    def __init__(self, frame_size: Tuple[int, int], params: SMDParams):
        self.params = params
        w, h = frame_size
        self._frame_size = (w, h)

        # фоновые модели
        self.bgL = DualBGModel((h, w))
        self.bgR = DualBGModel((h, w))

        # персистентность
        self.persist_k = int(getattr(self.params, "persist_k", 4))
        self.persist_m = int(getattr(self.params, "persist_m", 3))
        self.histL = deque(maxlen=self.persist_k)
        self.histR = deque(maxlen=self.persist_k)

        # анти-дрейф
        self._drift_streak = 0
        self.is_night = False

        # ROI (если задана)
        self.roi: Optional[np.ndarray] = None
        roi_path = getattr(self.params, "roi_mask", None)
        if roi_path:
            try:
                m = cv.imread(roi_path, cv.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[:2] != (h, w):
                        m = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
                    self.roi = (m > 0).astype(np.uint8) * 255
            except Exception:
                self.roi = None

    def _reset_background(self):
        w, h = self._frame_size
        try:
            self.bgL = DualBGModel((h, w))
            self.bgR = DualBGModel((h, w))
        except Exception:
            pass
        self.histL = deque(maxlen=self.persist_k)
        self.histR = deque(maxlen=self.persist_k)

    def step(self, rectL_bgr: np.ndarray, rectR_bgr: np.ndarray):
        gL = as_gray(rectL_bgr)
        gR = as_gray(rectR_bgr)

        # день/ночь
        night_auto = bool(getattr(self.params, "night_auto", True))
        night_thr = float(getattr(self.params, "night_luma_thr", 50.0))
        self.is_night = (float(np.mean(gL)) < night_thr) if night_auto else False

        gL = _vr_pre(gL, self.is_night)
        gR = _vr_pre(gR, self.is_night)

        use_clahe = bool(getattr(self.params, "use_clahe", True))
        if self.is_night:
            use_clahe = False  # ночью лучше без CLAHE → меньше фантомов

        # базовые маски движения
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

        # подавляем «медленный» дрейф облаков В НЕБЕ
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
            h2, _ = gL.shape[:2]
            split = float(getattr(self.params, "y_area_split", 0.55))
            sky_y = int(split * h2)
            sky_mask = np.zeros_like(mL, dtype=np.uint8); sky_mask[:sky_y, :] = 255
            not_sky = cv.bitwise_not(sky_mask)

            mL = cv.bitwise_or(cv.bitwise_and(mL, not_sky), cv.bitwise_and(mL_slow, sky_mask))
            mR = cv.bitwise_or(cv.bitwise_and(mR, not_sky), cv.bitwise_and(mR_slow, sky_mask))
        except Exception:
            pass

        # подрезка сверху (если нужно)
        crop_top = int(getattr(self.params, "crop_top", 0))
        if crop_top > 0:
            mL[:crop_top, :] = 0
            mR[:crop_top, :] = 0

        # --- ROI: один блок, бинаризация + автоподгон под текущий размер масок ---
        if self.roi is not None:
            if self.roi.ndim == 3:
                self.roi = cv.cvtColor(self.roi, cv.COLOR_BGR2GRAY)
            self.roi = (self.roi > 0).astype(np.uint8) * 255
            h_cur, w_cur = mL.shape[:2]
            if self.roi.shape[:2] != (h_cur, w_cur):
                self.roi = cv.resize(self.roi, (w_cur, h_cur), interpolation=cv.INTER_NEAREST)
                self.roi = (self.roi > 0).astype(np.uint8) * 255
            mL = cv.bitwise_and(mL, self.roi)
            mR = cv.bitwise_and(mR, self.roi)
        # --------------------------------------------------------------------------

        # ночная коррекция min_area
        min_area = int(getattr(self.params, "min_area", 25))
        if self.is_night:
            mult = float(getattr(self.params, "min_area_night_mult", 4.0))
            min_area = int(min_area * mult)
        max_area = int(getattr(self.params, "max_area", 0))

        # подчистка по компонентам
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

        # персистентность
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
        mL_final = _persist_mask(self.histL, mL_clean, self.persist_m)
        mR_final = _persist_mask(self.histR, mR_clean, self.persist_m)

        # анти-дрейф (перезапуск фона при «заливке» кадра)
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

        # детекции -> боксы
        boxesL = _boxes_from_mask(mL_final, min_area=min_area, max_area=max_area)
        boxesR = _boxes_from_mask(mR_final, min_area=min_area, max_area=max_area)

        # НЕБО: отсекаем широкие облачные полосы
        split = float(getattr(self.params, "y_area_split", 0.55))
        boxesL = _shape_gate_sky(boxesL, h, split)
        boxesR = _shape_gate_sky(boxesR, h, split)

        # === SAILS-ONLY: включается, если есть ROI воды ИЛИ передан флаг --sails_only ===
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

        # спаривание L/R и досведение NCC
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
