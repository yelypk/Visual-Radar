
from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv
from .utils import BBox, to_bbox, clamp

def ncc(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    am = a - a.mean(); bm = b - b.mean()
    denom = (np.linalg.norm(am)*np.linalg.norm(bm) + 1e-6)
    return float((am*bm).sum()/denom)

def match_template_ncc(search_img, templ):
    res = cv.matchTemplate(search_img, templ, cv.TM_CCOEFF_NORMED)
    minv, maxv, minloc, maxloc = cv.minMaxLoc(res)
    return maxv, maxloc

def gate_pairs_rectified(boxesL: List[BBox], boxesR: List[BBox], y_eps:int, dmin:int, dmax:int) -> List[Tuple[int,int]]:
    pairs = []
    for i,bl in enumerate(boxesL):
        best_j = -1; best_cost = 1e9
        for j,br in enumerate(boxesR):
            if abs(bl.cy - br.cy) > y_eps: 
                continue
            disp = abs(bl.cx - br.cx)
            if disp < dmin or disp > dmax: 
                continue
            cost = abs(bl.cy - br.cy) + 0.1*abs(disp - (dmin+dmax)/2.0)
            if cost < best_cost:
                best_cost = cost; best_j = j
        if best_j >= 0:
            pairs.append((i,best_j))
    return pairs

def epipolar_ncc_match(rectL_gray, rectR_gray, boxL: BBox, search_pad:int, patch:int, ncc_min:float) -> Optional[BBox]:
    h,w = rectL_gray.shape[:2]
    half = patch//2
    x0 = int(clamp(boxL.cx - half, 0, w-1))
    y0 = int(clamp(boxL.cy - half, 0, h-1))
    x1 = int(clamp(x0 + patch, 0, w)); y1 = int(clamp(y0 + patch, 0, h))
    templ = rectL_gray[y0:y1, x0:x1]
    if templ.shape[0] < patch//2 or templ.shape[1] < patch//2:
        return None

    sx0 = int(clamp(boxL.cx - search_pad, 0, w-1))
    sx1 = int(clamp(boxL.cx + search_pad, 0, w))
    sy0 = int(clamp(boxL.cy - search_pad//6, 0, h-1))
    sy1 = int(clamp(boxL.cy + search_pad//6, 0, h))
    search = rectR_gray[sy0:sy1, sx0:sx1]
    if search.shape[0] < templ.shape[0] or search.shape[1] < templ.shape[1]:
        return None

    score, loc = match_template_ncc(search, templ)
    if score < ncc_min:
        return None
    px,py = loc
    rx = sx0 + px; ry = sy0 + py
    rb = to_bbox(rx, ry, templ.shape[1], templ.shape[0])
    return rb
