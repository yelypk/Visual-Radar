from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2 as cv

from visual_radar.utils import BBox

# -----------------------------
# Helpers
# -----------------------------
def as_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB/BGR image to GRAY uint8 (required by most OpenCV algorithms).
    """
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
    Select structuring element size based on area.
    min_area ~ k^2 → k ~ sqrt(min_area).
    """
    k = int(max(3, np.sqrt(max(1, float(min_area))) * 0.50 * float(boost)))
    # Make odd
    if k % 2 == 0:
        k += 1
    k = int(np.clip(k, 3, 21))  # reasonable limits
    return k, k


def _morph_clean(mask: np.ndarray, min_area: int, size_aware_morph: bool = True) -> np.ndarray:
    """
    Morphological cleaning of mask.
    """
    if mask is None or mask.size == 0:
        return mask
    mask = (mask > 0).astype(np.uint8) * 255
    if size_aware_morph:
        kx, ky = _kernel_from_area(min_area)
    else:
        kx = ky = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kx, ky))
    # open -> remove noise; close to connect
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    return mask


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
    Two background models:
      - fast: short memory → detects fast motion (sails, birds, waves)
      - slow: long memory → detects slow drift (clouds/general brightness)
    Implemented with MOG2 for stability on Win/FFmpeg.
    """
    def __init__(self, shape_hw: Tuple[int, int]):
        h, w = int(shape_hw[0]), int(shape_hw[1])
        # Practical parameters; history/varThreshold are flexible per docs
        self.fast = cv.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
        self.slow = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=12, detectShadows=False)

    def apply_fast(self, gray: np.ndarray) -> np.ndarray:
        fg = self.fast.apply(gray, learningRate=-1)  # auto LR
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
    Main function: get motion mask and list of bounding boxes.
    - Optional CLAHE (robust to uneven exposure).
    - Two masks from "fast" and "slow" models.
    - Weighted combination, morphological cleaning, area filtering.
    """
    g = gray
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    mf = bg.apply_fast(g)
    ms = bg.apply_slow(g)

    # Threshold coefficients — like "weights" for mask contributions:
    alpha = float(max(0.0, thr_fast))
    beta = float(max(0.0, thr_slow))
    denom = max(1e-6, alpha + beta)
    alpha /= denom
    beta /= denom

    mix = cv.addWeighted(mf, alpha, ms, beta, 0.0)
    mix = (mix > 0).astype(np.uint8) * 255

    # Morphological cleaning
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
    Return two masks:
      - static: conservative (almost-static background)
      - slow: slow drift (clouds etc.), to "soften" the top of the frame
    """
    g = gray
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    mf = bg.apply_fast(g)
    ms = bg.apply_slow(g)

    k = max(3, int(kernel_size) | 1)
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

    fast_bin = (mf > 0).astype(np.uint8) * 255
    slow_bin = (ms > 0).astype(np.uint8) * 255

    static = cv.bitwise_not(fast_bin)
    static = cv.morphologyEx(static, cv.MORPH_OPEN, ker, iterations=1)

    slow = cv.morphologyEx(slow_bin, cv.MORPH_OPEN, ker, iterations=1)
    return static, slow
