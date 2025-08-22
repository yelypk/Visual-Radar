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
    def __init__(self, frame_size: Tuple[int, int], params: SMDParams):
        self.params = params
        w, h = frame_size
        # запоминаем исходный размер кадра (w, h)
        self._frame_size = (w, h)

        # фоновые модели
        self.bgL = DualBGModel((h, w))
        self.bgR = DualBGModel((h, w))

        # temporal persistence
        self.persist_k = int(getattr(self.params, 'persist_k', 4))  # окно
        self.persist_m = int(getattr(self.params, 'persist_m', 3))  # минимум попаданий
        self.histL = deque(maxlen=self.persist_k)
        self.histR = deque(maxlen=self.persist_k)

        # защита от дрейфа
        self._drift_streak = 0
        self._last_fg_ratio = 0.0

        # ROI-маска (опционально)
        self.roi = None
        roi_path = getattr(self.params, 'roi_mask', None)
        if roi_path:
            try:
                m = cv.imread(roi_path, cv.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[:2] != (h, w):
                        m = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
                    self.roi = (m > 0).astype(np.uint8) * 255
            except Exception:
                self.roi = None

        self.is_night = False

    def _reset_background(self):
        """Полный сброс фона и истории персистентности (анти-дрейф)."""
        w, h = self._frame_size
        try:
            self.bgL = DualBGModel((h, w))
            self.bgR = DualBGModel((h, w))
        except Exception:
            pass
        # очистка истории персистентности
        self.histL = deque(maxlen=self.persist_k)
        self.histR = deque(maxlen=self.persist_k)

    def step(self, rectL_bgr, rectR_bgr):
        gL = as_gray(rectL_bgr)
        gR = as_gray(rectR_bgr)

        night_auto = bool(getattr(self.params, "night_auto", True))
        night_thr = float(getattr(self.params, "night_luma_thr", 50.0))
        is_night = (float(np.mean(gL)) < night_thr) if night_auto else False
        self.is_night = is_night

        gL = _vr_pre(gL, is_night)
        gR = _vr_pre(gR, is_night)

        use_clahe = bool(getattr(self.params, 'use_clahe', True))
        if is_night:
            use_clahe = False

        mL, _ = find_motion_bboxes(
            gL, self.bgL,
            self.params.min_area, self.params.max_area,
            self.params.thr_fast, self.params.thr_slow,
            use_clahe=use_clahe,
            size_aware_morph=self.params.size_aware_morph
        )
        mR, _ = find_motion_bboxes(
            gR, self.bgR,
            self.params.min_area, self.params.max_area,
            self.params.thr_fast, self.params.thr_slow,
            use_clahe=use_clahe,
            size_aware_morph=self.params.size_aware_morph
        )

        # отсечём верх (например, небо/блики)
        crop_top = int(getattr(self.params, 'crop_top', 0))
        if crop_top > 0:
            mL[:crop_top, :] = 0
            mR[:crop_top, :] = 0

        # ROI-маска (белое — анализируем, чёрное — игнорируем)
        if self.roi is not None:
            mL = cv.bitwise_and(mL, self.roi)
            mR = cv.bitwise_and(mR, self.roi)

        # адаптивный min_area для ночи
        min_area = int(getattr(self.params, "min_area", 25))
        if is_night:
            min_area = int(min_area * float(getattr(self.params, "min_area_night_mult", 4.0)))
        max_area = int(getattr(self.params, "max_area", 0))

        # подчистим мелочь по компонентам
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

        # персистентность
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

        # === анти-дрейф: авто-сброс фона при «заливке» FG ===
        h, w = gL.shape[:2]
        fg_ratio_L = float(np.count_nonzero(mL_final)) / float(h * w)
        fg_ratio_R = float(np.count_nonzero(mR_final)) / float(h * w)
        fg_ratio = max(fg_ratio_L, fg_ratio_R)  # можно взять max или среднее; берём max — консервативно
        self._last_fg_ratio = fg_ratio

        drift_max_fg = float(getattr(self.params, 'drift_max_fg_pct', 0.25))   # 25% по умолчанию
        drift_max_len = int(getattr(self.params, 'drift_max_frames', 60))      # 60 кадров подряд

        if fg_ratio > drift_max_fg:
            self._drift_streak += 1
        else:
            self._drift_streak = 0

        if self._drift_streak >= drift_max_len:
            # сбрасываем фоновые модели и историю
            self._reset_background()
            self._drift_streak = 0
            # необязательно: можно вывести сообщение в лог
            # print("[drift] background reset (fg_ratio=%.2f)" % fg_ratio)

        # детекции по маскам (после персистентности)
        boxesL = _boxes_from_mask(mL_final, min_area=min_area, max_area=max_area)
        boxesR = _boxes_from_mask(mR_final, min_area=min_area, max_area=max_area)

        # быстрый геометрический гейтинг
        pairs = gate_pairs_rectified(boxesL, boxesR, self.params.y_eps, self.params.dmin, self.params.dmax)

        # NCC-досведение неподтверждённых левых боксов
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