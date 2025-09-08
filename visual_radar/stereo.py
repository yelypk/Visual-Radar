from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv

from visual_radar.utils import BBox

try:
    from ._ncc import epipolar_match as _ncc_match  # (gL, gR, cx, cy, patch, pad) -> (xr, score)
except Exception:
    _ncc_match = None


def gate_pairs_rectified(
    boxesL: List[BBox],
    boxesR: List[BBox],
    y_eps: float = 6.0,
    dmin: float = -512.0,
    dmax: float = 512.0,
    w_y: float = 1.0,
    w_d: float = 0.25,
) -> List[Tuple[int, int]]:
    if not boxesL or not boxesR:
        return []

    candidates: List[Tuple[int, int, float]] = []
    for i, bl in enumerate(boxesL):
        for j, br in enumerate(boxesR):
            dy = float(abs(bl.cy - br.cy))
            if dy > float(y_eps):
                continue
            d = float(bl.cx - br.cx)
            if d < float(dmin) or d > float(dmax):
                continue
            score = float(w_y) * dy + float(w_d) * abs(d)
            candidates.append((i, j, score))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[2])
    usedL, usedR, pairs = set(), set(), []
    for i, j, _ in candidates:
        if i in usedL or j in usedR:
            continue
        usedL.add(i)
        usedR.add(j)
        pairs.append((i, j))
    return pairs


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def _grad_img(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Безопасный градиент: Sobel поддерживает только ksize ∈ {1,3,5,7}.
    Любое другое значение приводим к ближайшему допустимому.
    """
    k = int(ksize)
    if   k <= 1: k = 1
    elif k <= 3: k = 3
    elif k <= 5: k = 5
    elif k <= 7: k = 7
    else:        k = 7

    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=k)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=k)
    return cv.magnitude(gx, gy)


def _crop_patch(img: np.ndarray, cx: float, cy: float, half: int) -> Optional[np.ndarray]:
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


def _parabolic_subpixel_1d(vals: np.ndarray) -> float:
    if vals.shape[0] != 3:
        return 0.0
    l, c, r = float(vals[0]), float(vals[1]), float(vals[2])
    denom = (l - 2.0 * c + r)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (l - r) / denom


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
    gL = _to_gray(grayL)
    gR = _to_gray(grayR)

    if _ncc_match is not None and not use_gradient:
        cx = int(round(boxL.cx)); cy = int(round(boxL.cy))
        xr, score = _ncc_match(gL, gR, cx, cy, int(patch_size), int(search_pad))
        if float(score) < float(ncc_min):
            return None
        return BBox(
            float(xr - boxL.w * 0.5),
            float(boxL.cy - boxL.h * 0.5),
            float(boxL.w),
            float(boxL.h),
        )

    if use_gradient:
        srcL = _grad_img(gL, ksize=int(grad_ksize))
        srcR = _grad_img(gR, ksize=int(grad_ksize))
    else:
        srcL = gL.astype(np.float32)
        srcR = gR.astype(np.float32)

    half = max(2, int(patch_size) // 2)
    tpl = _crop_patch(srcL, boxL.cx, boxL.cy, half)
    if tpl is None:
        return None

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

    _min_val, max_val, _min_loc, max_loc = cv.minMaxLoc(res)
    if float(max_val) < float(ncc_min):
        return None

    dx_sub = 0.0
    if subpixel and res.shape[1] >= 3:
        cx_i = max_loc[0]
        if 1 <= cx_i < res.shape[1] - 1:
            triplet = res[max_loc[1], cx_i - 1: cx_i + 2].astype(np.float32)
            dx_sub = float(np.clip(_parabolic_subpixel_1d(triplet), -0.5, 0.5))

    top_left = (x1 + max_loc[0], y1 + max_loc[1])
    cxR = top_left[0] + tpl.shape[1] * 0.5 + dx_sub
    cyR = top_left[1] + tpl.shape[0] * 0.5

    return BBox(
        float(cxR - boxL.w * 0.5),
        float(cyR - boxL.h * 0.5),
        float(boxL.w),
        float(boxL.h),
    )


def _back_match_R_to_L(
    grayL: np.ndarray,
    grayR: np.ndarray,
    boxR: BBox,
    search_pad: int,
    patch_size: int,
    ncc_min: float,
    use_gradient: bool,
    grad_ksize: int,
    subpixel: bool,
) -> Optional[BBox]:
    return epipolar_ncc_match(
        grayL=grayR,
        grayR=grayL,
        boxL=boxR,
        search_pad=search_pad,
        patch_size=patch_size,
        ncc_min=ncc_min,
        use_gradient=use_gradient,
        grad_ksize=grad_ksize,
        subpixel=subpixel,
    )


def refine_pairs_lr_consistency(
    boxesL: List[BBox],
    boxesR: List[BBox],
    pairs: List[Tuple[int, int]],
    grayL: np.ndarray,
    grayR: np.ndarray,
    *,
    backtrack_pad: int = 8,
    patch_size: int = 13,
    ncc_min: float = 0.25,
    use_gradient: bool = True,
    grad_ksize: int = 3,
    subpixel: bool = True,
    max_cx_err_px: float = 2.0,
) -> List[Tuple[int, int]]:
    if not pairs:
        return pairs

    kept: List[Tuple[int, int]] = []
    gL = _to_gray(grayL)
    gR = _to_gray(grayR)

    for iL, iR in pairs:
        bL = boxesL[iL]; bR = boxesR[iR]
        back = _back_match_R_to_L(
            gL, gR, bR,
            search_pad=int(backtrack_pad),
            patch_size=int(patch_size),
            ncc_min=float(ncc_min),
            use_gradient=bool(use_gradient),
            grad_ksize=int(grad_ksize),
            subpixel=bool(subpixel),
        )
        if back is None:
            continue
        if abs(float(back.cx) - float(bL.cx)) <= float(max_cx_err_px):
            kept.append((iL, iR))
    return kept


def gate_and_refine_pairs(
    boxesL: List[BBox],
    boxesR: List[BBox],
    grayL: np.ndarray,
    grayR: np.ndarray,
    *,
    y_eps: float = 6.0, dmin: float = -512.0, dmax: float = 512.0, w_y: float = 1.0, w_d: float = 0.25,
    enable_lr: bool = True,
    backtrack_pad: int = 8, patch_size: int = 13, ncc_min: float = 0.25,
    use_gradient: bool = True, grad_ksize: int = 3, subpixel: bool = True,
    max_cx_err_px: float = 2.0,
) -> List[Tuple[int, int]]:
    pairs = gate_pairs_rectified(boxesL, boxesR, y_eps, dmin, dmax, w_y, w_d)
    if not enable_lr or not pairs:
        return pairs
    return refine_pairs_lr_consistency(
        boxesL, boxesR, pairs, grayL, grayR,
        backtrack_pad=backtrack_pad, patch_size=patch_size, ncc_min=ncc_min,
        use_gradient=use_gradient, grad_ksize=grad_ksize, subpixel=subpixel,
        max_cx_err_px=max_cx_err_px,
    )
