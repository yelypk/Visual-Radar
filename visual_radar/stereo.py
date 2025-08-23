# visual_radar/stereo.py
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv

from visual_radar.utils import BBox


def gate_pairs_rectified(
    boxesL: List[BBox],
    boxesR: List[BBox],
    y_eps: float = 6.0,
    dmin: float = -512.0,
    dmax: float = 512.0,
) -> List[Tuple[int, int]]:
    """
    Жадное сопоставление L/R-боксов на РЕКТИФИЦИРОВАННЫХ кадрах.
    Условие: |cyL - cyR| <= y_eps и disparity d = (cxL - cxR) в [dmin, dmax].
    Возвращает пары индексов (iL, iR).
    """
    if not boxesL or not boxesR:
        return []

    # Кандидаты: список (iL, iR, score), где score = |разность центров по X|
    cands: List[Tuple[int, int, float]] = []
    for i, bl in enumerate(boxesL):
        for j, br in enumerate(boxesR):
            if abs(bl.cy - br.cy) > float(y_eps):
                continue
            d = float(bl.cx - br.cx)
            if d < float(dmin) or d > float(dmax):
                continue
            score = abs(d)  # чем меньше модуль диспаритета, тем "плотнее" пара
            cands.append((i, j, score))

    if not cands:
        return []

    # Жадно берём лучшие соответствия, не допуская повторов индексов
    cands.sort(key=lambda t: t[2])  # по возрастанию |d|
    usedL = set()
    usedR = set()
    pairs: List[Tuple[int, int]] = []
    for i, j, _ in cands:
        if i in usedL or j in usedR:
            continue
        usedL.add(i)
        usedR.add(j)
        pairs.append((i, j))
    return pairs


def _crop_patch(img: np.ndarray, cx: float, cy: float, half: int) -> Optional[np.ndarray]:
    """Вырезать квадратный патч с центром (cx, cy) и половинкой размера half."""
    h, w = img.shape[:2]
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + 2 * half + 1
    y2 = y1 + 2 * half + 1
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return None
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img[y1:y2, x1:x2]


def epipolar_ncc_match(
    grayL: np.ndarray,
    grayR: np.ndarray,
    boxL: BBox,
    search_pad: int = 64,
    patch_size: int = 13,
    ncc_min: float = 0.25,
) -> Optional[BBox]:
    """
    Поиск соответствия для левого бокса вдоль эпиполярной строки (горизонтали) в правом кадре
    с помощью NCC (cv.matchTemplate, TM_CCOEFF_NORMED).

    Возвращает правый BBox (того же размера, что и левый), либо None при неудаче.
    """
    gL = grayL if grayL.ndim == 2 else cv.cvtColor(grayL, cv.COLOR_BGR2GRAY)
    gR = grayR if grayR.ndim == 2 else cv.cvtColor(grayR, cv.COLOR_BGR2GRAY)

    half = max(2, int(patch_size) // 2)
    tpl = _crop_patch(gL, boxL.cx, boxL.cy, half)
    if tpl is None:
        return None

    h, w = gR.shape[:2]
    # Поиск в горизонтальном окне вокруг cx (± search_pad) с небольшой свободы по Y (± 2)
    x1 = max(0, int(round(boxL.cx)) - int(search_pad))
    x2 = min(w, int(round(boxL.cx)) + int(search_pad))
    y = int(round(boxL.cy))
    y1 = max(0, y - 2)
    y2 = min(h, y + 3)

    if x2 - x1 < tpl.shape[1] + 2 or y2 - y1 < tpl.shape[0] + 2:
        # слишком узкое окно поиска
        return None

    roi = gR[y1:y2, x1:x2]
    try:
        res = cv.matchTemplate(roi, tpl, cv.TM_CCOEFF_NORMED)
    except cv.error:
        return None

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if float(max_val) < float(ncc_min):
        return None

    # координаты в правом кадре
    top_left = (x1 + max_loc[0], y1 + max_loc[1])
    cxR = top_left[0] + tpl.shape[1] * 0.5
    cyR = top_left[1] + tpl.shape[0] * 0.5

    # Возвращаем бокс того же размера, что и левый
    return BBox(float(cxR - boxL.w * 0.5),
                float(cyR - boxL.h * 0.5),
                float(boxL.w),
                float(boxL.h))
