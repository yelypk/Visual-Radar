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
    w_y: float = 1.0,
    w_d: float = 0.25,
) -> List[Tuple[int, int]]:
    """
    Жадное сопоставление L/R-боксов на РЕКТИФИЦИРОВАННЫХ кадрах.
    Условие кандидата: |cyL - cyR| <= y_eps и disparity d = (cxL - cxR) в [dmin, dmax].
    Выбираем пары, минимизируя комбинированный скор: w_y*|Δy| + w_d*|d|.
    """
    if not boxesL or not boxesR:
        return []

    cands: List[Tuple[int, int, float]] = []
    for i, bl in enumerate(boxesL):
        for j, br in enumerate(boxesR):
            dy = float(abs(bl.cy - br.cy))
            if dy > float(y_eps):
                continue
            d = float(bl.cx - br.cx)
            if d < float(dmin) or d > float(dmax):
                continue
            score = float(w_y) * dy + float(w_d) * abs(d)
            cands.append((i, j, score))

    if not cands:
        return []

    cands.sort(key=lambda t: t[2])  # чем меньше скор, тем лучше
    usedL, usedR, pairs = set(), set(), []
    for i, j, _ in cands:
        if i in usedL or j in usedR:
            continue
        usedL.add(i); usedR.add(j)
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


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def _grad_img(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Модуль градиента (Sobel) → float32, для более устойчивого NCC."""
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=ksize)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=ksize)
    mag = cv.magnitude(gx, gy)  # float32
    return mag


def _parabolic_subpixel_1d(vals: np.ndarray) -> float:
    """
    Субпиксельная поправка по трём точкам (l, c, r).
    Возвращает dx в [-0.5, 0.5] относительно центра.
    """
    if vals.shape[0] != 3:
        return 0.0
    l, c, r = float(vals[0]), float(vals[1]), float(vals[2])
    denom = (l - 2.0 * c + r)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (l - r) / denom  # стандартная формула для параболы


def epipolar_ncc_match(
    grayL: np.ndarray,
    grayR: np.ndarray,
    boxL: BBox,
    search_pad: int = 64,
    patch_size: int = 13,
    ncc_min: float = 0.25,
    use_gradient: bool = True,
    grad_ksize: int = 3,
    subpixel: bool = True,
) -> Optional[BBox]:
    """
    Поиск соответствия для левого бокса вдоль эпиполярной строки (горизонтали) в правом кадре
    с помощью NCC (cv.matchTemplate, TM_CCOEFF_NORMED), опционально по градиенту.
    Возвращает правый BBox (того же размера, что и левый), либо None при неудаче.
    """
    gL = _to_gray(grayL)
    gR = _to_gray(grayR)

    # по градиенту — устойчивее к изменениям яркости
    if use_gradient:
        gL32 = _grad_img(gL, ksize=int(grad_ksize))
        gR32 = _grad_img(gR, ksize=int(grad_ksize))
        srcL, srcR = gL32, gR32
    else:
        # matchTemplate ок c 8U/32F → перейдём на 32F
        srcL = gL.astype(np.float32)
        srcR = gR.astype(np.float32)

    half = max(2, int(patch_size) // 2)
    tpl = _crop_patch(srcL, boxL.cx, boxL.cy, half)
    if tpl is None:
        return None

    # фильтр малотекстурных патчей (иначе NCC сильно шумит)
    if float(np.var(tpl)) < 5.0:
        return None

    h, w = srcR.shape[:2]
    x1 = max(0, int(round(boxL.cx)) - int(search_pad))
    x2 = min(w, int(round(boxL.cx)) + int(search_pad))
    y = int(round(boxL.cy))
    y1 = max(0, y - 2)
    y2 = min(h, y + 3)

    if x2 - x1 < tpl.shape[1] + 2 or y2 - y1 < tpl.shape[0] + 2:
        return None

    roi = srcR[y1:y2, x1:x2]

    try:
        res = cv.matchTemplate(roi, tpl, cv.TM_CCOEFF_NORMED)
    except cv.error:
        return None

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if float(max_val) < float(ncc_min):
        return None

    # субпиксельная подстройка по X
    dx_sub = 0.0
    if subpixel and res.shape[1] >= 3:
        cx = max_loc[0]
        if 1 <= cx < res.shape[1] - 1:
            triplet = res[max_loc[1], cx - 1 : cx + 2].astype(np.float32)
            dx_sub = float(np.clip(_parabolic_subpixel_1d(triplet), -0.5, 0.5))

    # координаты в правом кадре
    top_left = (x1 + max_loc[0], y1 + max_loc[1])
    cxR = top_left[0] + tpl.shape[1] * 0.5 + dx_sub
    cyR = top_left[1] + tpl.shape[0] * 0.5

    return BBox(float(cxR - boxL.w * 0.5),
                float(cyR - boxL.h * 0.5),
                float(boxL.w),
                float(boxL.h))
