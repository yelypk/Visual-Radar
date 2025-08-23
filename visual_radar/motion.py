# visual_radar/motion.py
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2 as cv

from visual_radar.utils import BBox


# -----------------------------
# helpers
# -----------------------------
def as_gray(img) -> np.ndarray:
    """RGB/BGR -> GRAY uint8."""
    if img is None:
        raise ValueError("as_gray: None image")
    if len(img.shape) == 2:
        g = img
    else:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return g


def _kernel_from_area(min_area: int, boost: float = 1.0) -> Tuple[int, int]:
    """
    Подбираем размер структурирующего элемента по площади.
    min_area ~ k^2 → k ~ sqrt(min_area).
    """
    k = int(max(3, np.sqrt(max(1, float(min_area))) * 0.50 * float(boost)))
    # делаем нечётным
    if k % 2 == 0:
        k += 1
    k = int(np.clip(k, 3, 21))  # разумные пределы
    return k, k


def _morph_clean(mask: np.ndarray, min_area: int, size_aware_morph: bool = True) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    mask = (mask > 0).astype(np.uint8) * 255
    if size_aware_morph:
        kx, ky = _kernel_from_area(min_area)
    else:
        kx = ky = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kx, ky))
    # open -> remove noise; close a bit to connect
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _boxes_from_mask(mask: np.ndarray, min_area: int, max_area: int) -> List[BBox]:
    boxes: List[BBox] = []
    if mask is None or mask.size == 0:
        return boxes
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        area = int(w) * int(h)
        if area < max(1, int(min_area)):
            continue
        if max_area and area > int(max_area):
            continue
        boxes.append(BBox(float(x), float(y), float(w), float(h)))
    return boxes


# -----------------------------
# Background models
# -----------------------------
class DualBGModel:
    """
    Две фоновые модели:
      - fast: короткая память → выделяет быстрое движение (паруса, птицы, волны)
      - slow: длинная память → улавливает медленный дрейф (облака/общая яркость)
    Реализация на MOG2, т.к. он стабильно работает на Win/FFmpeg.
    """
    def __init__(self, shape_hw: Tuple[int, int]):
        h, w = int(shape_hw[0]), int(shape_hw[1])
        # Параметры подобраны практично; при желании можно прокинуть в SMDParams
        self.fast = cv.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
        self.slow = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=12, detectShadows=False)

    def apply_fast(self, gray: np.ndarray) -> np.ndarray:
        fg = self.fast.apply(gray, learningRate=-1)  # авто LR
        return (fg > 0).astype(np.uint8) * 255

    def apply_slow(self, gray: np.ndarray) -> np.ndarray:
        fg = self.slow.apply(gray, learningRate=-1)
        return (fg > 0).astype(np.uint8) * 255


# -----------------------------
# Motion masks
# -----------------------------
def find_motion_bboxes(
    gray: np.ndarray,
    bg: DualBGModel,
    min_area: int,
    max_area: int,
    thr_fast: float,
    thr_slow: float,
    use_clahe: bool = True,
    size_aware_morph: bool = True,
) -> Tuple[np.ndarray, List[BBox]]:
    """
    Главная функция: получить маску движения и список боксов.
    - Делаем (опционально) CLAHE для устойчивости к неравномерной экспозиции.
    - Получаем две маски от "быстрой" и "медленной" модели.
    - Усиливаем, чистим морфологией, фильтруем по площади.
    Возвращает (mask, boxes). В большинстве мест дальше используется только mask.
    """
    g = gray
    if use_clahe:
        # CLAHE помогает на воде с бликами и ползущим градиентом в небе
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    mf = bg.apply_fast(g)
    ms = bg.apply_slow(g)

    # Пороговые коэффициенты можно использовать как "вес" вкладов масок:
    alpha = float(max(0.0, thr_fast))
    beta = float(max(0.0, thr_slow))
    # нормализуем веса в [0..1], чтобы не взрывать значения
    denom = max(1e-6, alpha + beta)
    alpha /= denom
    beta /= denom

    mix = cv.addWeighted(mf, alpha, ms, beta, 0.0)
    mix = (mix > 0).astype(np.uint8) * 255

    # Очистка по морфологии
    mask = _morph_clean(mix, int(min_area), bool(size_aware_morph))
    boxes = _boxes_from_mask(mask, int(min_area), int(max_area))
    return mask, boxes


def make_masks_static_and_slow(
    gray: np.ndarray,
    bg: DualBGModel,
    thr_fast: float,
    thr_slow: float,
    use_clahe: bool = True,
    kernel_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вернуть две маски:
      - static: консервативная (почти-статичный фон) — может использоваться для подавления шума
      - slow: медленный дрейф (облака и т.п.), чтобы "смягчить" верхнюю часть кадра
    На практике в проекте используется в небе: заменяем там fast-маску на медленную.
    """
    g = gray
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    mf = bg.apply_fast(g)
    ms = bg.apply_slow(g)

    # "static" — всё, что НЕ отмечено как быстро меняющееся (инверт fast),
    # но оставим только те области, где slow тоже спокоен → уменьшаем фантомы.
    k = max(3, int(kernel_size) | 1)
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

    fast_bin = (mf > 0).astype(np.uint8) * 255
    slow_bin = (ms > 0).astype(np.uint8) * 255

    static = cv.bitwise_not(fast_bin)
    static = cv.morphologyEx(static, cv.MORPH_OPEN, ker, iterations=1)

    slow = cv.morphologyEx(slow_bin, cv.MORPH_OPEN, ker, iterations=1)
    return static, slow

