from __future__ import annotations

from typing import Iterable, List, Tuple
import cv2 as cv
import numpy as np

from visual_radar.utils import BBox

def draw_boxes(
    img: np.ndarray,
    boxes: Iterable[BBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    tag: str = "",
    thickness: int = 2
) -> None:
    """
    Draw bounding boxes with optional label.
    """
    if img is None:
        return
    for b in boxes:
        x, y, w, h = b.to_int()
        cv.rectangle(img, (x, y), (x + w, y + h), color, thickness, cv.LINE_AA)
    if tag:
        cv.putText(img, tag, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

def stack_lr(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """
    Horizontally stack two images, aligning heights without distortion.
    """
    if left_bgr is None or right_bgr is None:
        return left_bgr if right_bgr is None else right_bgr
    hL, wL = left_bgr.shape[:2]
    hR, wR = right_bgr.shape[:2]
    if hL != hR:
        scale = hL / float(hR)
        right_bgr = cv.resize(right_bgr, (int(round(wR * scale)), hL), interpolation=cv.INTER_AREA)
    return np.hstack([left_bgr, right_bgr])

def imshow_resized(win: str, img: np.ndarray, max_w: int = 1920, max_h: int = 1080) -> None:
    """
    Show image in a window, resizing to fit the screen (WINDOW_NORMAL).
    """
    if img is None:
        return
    h, w = img.shape[:2]
    sx = min(1.0, max_w / float(w))
    sy = min(1.0, max_h / float(h))
    s = min(sx, sy)
    if s < 0.999:
        img = cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA)
    cv.imshow(win, img)

def draw_hud(
    img: np.ndarray,
    lines: List[str],
    corner: str = "tr",
    alpha: float = 0.55
) -> None:
    """
    Draw a semi-transparent HUD with text.
    corner: 'tl' | 'tr' | 'bl' | 'br'
    """
    if img is None or not lines:
        return
    pad, lh, fw = 8, 20, 2
    w = max(cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, 0.56, fw)[0][0] for t in lines) + pad * 2
    h = lh * len(lines) + pad * 2

    H, W = img.shape[:2]
    x = 0 if "l" in corner else W - w
    y = 0 if "t" in corner else H - h

    # Background rectangle
    roi = img[y:y + h, x:x + w].copy()
    overlay = roi.copy()
    cv.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 0), -1)
    cv.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    img[y:y + h, x:x + w] = roi

    # Text lines
    cy = y + pad + 14
    for t in lines:
        cv.putText(img, t, (x + pad, cy), cv.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv.LINE_AA)
        cy += lh

