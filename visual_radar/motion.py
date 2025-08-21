from typing import Tuple, List, Optional
import numpy as np
import cv2 as cv
from .utils import BBox, to_bbox, clamp

def as_gray(img):
    import cv2 as cv
    if img.ndim == 2:
        return img
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

class DualBGModel:
    """Dual-timescale running averages: fast and slow. Produces foreground mask."""
    def __init__(self, shape, alpha_fast=0.08, alpha_slow=0.005):
        self.fast = None
        self.slow = None
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.shape = shape

    def update(self, gray):
        g = gray.astype(np.float32)
        if self.fast is None:
            self.fast = g.copy()
            self.slow = g.copy()
        self.fast = (1-self.alpha_fast)*self.fast + self.alpha_fast*g
        self.slow = (1-self.alpha_slow)*self.slow + self.alpha_slow*g
        df = cv.absdiff(g, self.fast).astype(np.float32)
        ds = cv.absdiff(g, self.slow).astype(np.float32)
        return df, ds

def compute_mad(img, sample_fraction=0.1):
    h,w = img.shape[:2]
    step = max(1, int(1/np.sqrt(sample_fraction)))
    sample = img[::step, ::step].astype(np.float32)
    med = np.median(sample)
    mad = np.median(np.abs(sample - med)) + 1e-6
    return 1.4826 * mad  # ≈ sigma

def adaptive_threshold_from_noise(gray, user_thr_fast: Optional[float], user_thr_slow: Optional[float]):
    sigma = compute_mad(gray)
    base = 3.0 * sigma
    thr_fast = clamp(user_thr_fast if user_thr_fast is not None else base, 3.0, 20.0)
    thr_slow = clamp(user_thr_slow if user_thr_slow is not None else base*0.7, 2.0, 15.0)
    return thr_fast, thr_slow

def find_motion_bboxes(gray, bg: DualBGModel,
                       min_area:int, max_area:int,
                       thr_fast:Optional[float], thr_slow: Optional[float],
                       use_clahe=True,
                       size_aware_morph=True) -> Tuple[np.ndarray, List[BBox]]:
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(gray)
    else:
        g = gray

    tf, ts = adaptive_threshold_from_noise(g, thr_fast, thr_slow)
    df, ds = bg.update(g)
    fg = ((df > tf) | (ds > ts)).astype(np.uint8)*255

    if size_aware_morph:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel, iterations=1)
        fg_hat = cv.morphologyEx(g, cv.MORPH_TOPHAT, kernel, iterations=1)
        fg = cv.max(fg, (fg_hat > tf).astype(np.uint8)*255)
        fg = cv.dilate(fg, kernel, iterations=1)

    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes: List[BBox] = []
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
        area = w*h
        if area < min_area: continue
        if max_area > 0 and area > max_area: continue
        boxes.append(to_bbox(x,y,w,h))
    return fg, boxes
