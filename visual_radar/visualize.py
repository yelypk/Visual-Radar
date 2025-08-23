from typing import List
import numpy as np
import cv2 as cv

from visual_radar.utils import BBox  # явный импорт

def draw_boxes(img, boxes: List[BBox], color=(0, 255, 0), label: str = "") -> None:
    """Простая отрисовка боксов."""
    for b in boxes:
        x, y, w, h = map(int, (b.x, b.y, b.w, b.h))
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if label:
            cv.putText(
                img,
                label,
                (x, max(0, y - 5)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv.LINE_AA,
            )

def stack_lr(left, right):
    """Склейка левого/правого в один кадр по горизонтали."""
    return np.hstack([left, right])

def resize_for_display(img, max_w: int = 1280, max_h: int = 720):
    """Масштабирование под окно отображения без увеличения."""
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h), 1.0)
    if s < 1.0:
        return cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA)
    return img

def imshow_resized(win: str, img, max_w: int = 1280, max_h: int = 720) -> None:
    """Показ кадра с автоматическим ресайзом под экран."""
    cv.imshow(win, resize_for_display(img, max_w, max_h))

