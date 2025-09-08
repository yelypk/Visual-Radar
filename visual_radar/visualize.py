# visual_radar/visualize.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import cv2 as cv
import numpy as np

from visual_radar.utils import BBox


# ---------- окно показа: гарантируем WINDOW_NORMAL ----------
def _window_exists(win: str) -> bool:
    try:
        return cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) >= 0
    except Exception:
        return False

def _window_is_autosize(win: str) -> bool:
    try:
        return cv.getWindowProperty(win, cv.WND_PROP_AUTOSIZE) > 0.5
    except Exception:
        return False

def _create_normal_window(win: str) -> None:
    try:
        cv.namedWindow(win, cv.WINDOW_NORMAL | cv.WINDOW_GUI_EXPANDED)
    except Exception:
        cv.namedWindow(win, cv.WINDOW_NORMAL)

def _ensure_normal_window(win: str) -> None:
    """
    Если окно отсутствует — создаём NORMAL.
    Если окно уже есть, но AUTOSIZE — уничтожаем и создаём NORMAL.
    Это критично: OpenCV не меняет флаги уже созданного окна.
    """
    if not _window_exists(win):
        _create_normal_window(win)
        return
    # окно есть: если AUTOSIZE, пересоздаём
    if _window_is_autosize(win):
        cv.destroyWindow(win)
        _create_normal_window(win)


# ---------- утилиты рисования/склейки ----------
def _to_bgr(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def _match_heights(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if ha == hb:
        return a, b
    target_h = min(ha, hb)
    a = cv.resize(a, (int(round(wa * (target_h / ha))), target_h), interpolation=cv.INTER_AREA)
    b = cv.resize(b, (int(round(wb * (target_h / hb))), target_h), interpolation=cv.INTER_AREA)
    return a, b

def stack_lr(left_bgr: np.ndarray | None, right_bgr: np.ndarray | None) -> np.ndarray | None:
    L = _to_bgr(left_bgr)
    R = _to_bgr(right_bgr)
    if L is None and R is None:
        return None
    if L is None:
        return R
    if R is None:
        return L
    L, R = _match_heights(L, R)
    return np.hstack([L, R])

def draw_boxes(
    img: np.ndarray,
    boxes: Iterable[BBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    tag: str = "",
    thickness: int = 2
) -> None:
    if img is None:
        return
    for b in boxes:
        x, y, w, h = b.to_int()
        cv.rectangle(img, (x, y), (x + w, y + h), color, thickness, cv.LINE_AA)
    if tag:
        cv.putText(img, tag, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)


# ---------- показ с автоподгонкой под окно ----------
def imshow_resized(win: str, img: np.ndarray, max_w: int = 1920, max_h: int = 1080) -> None:
    """
    Показывает изображение в окне с гарантированным WINDOW_NORMAL,
    мягко уменьшая изображение до max_w×max_h (без апскейла).
    """
    if img is None:
        return

    # 1) гарантируем окно NORMAL (если уже AUTOSIZE — пересоздаём)
    _ensure_normal_window(win)

    # 2) уменьшаем только для показа
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h), 1.0)
    if s < 1.0:
        img = cv.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv.INTER_AREA)

    cv.imshow(win, img)


def draw_hud(
    img: np.ndarray,
    lines: List[str],
    corner: str = "tr",
    alpha: float = 0.55
) -> None:
    if img is None or not lines:
        return
    pad, lh, fw = 8, 20, 2
    w = max(cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, 0.56, fw)[0][0] for t in lines) + pad * 2
    h = lh * len(lines) + pad * 2

    H, W = img.shape[:2]
    x = 0 if "l" in corner else W - w
    y = 0 if "t" in corner else H - h

    roi = img[y:y + h, x:x + w].copy()
    overlay = roi.copy()
    cv.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 0), -1)
    cv.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    img[y:y + h, x:x + w] = roi

    cy = y + pad + 14
    for t in lines:
        cv.putText(img, t, (x + pad, cy), cv.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv.LINE_AA)
        cy += lh


# ---------- отладка эпиполярной геометрии ----------
def draw_epipolar_lines_for_pairs(
    imgL: np.ndarray,
    imgR: np.ndarray,
    boxesL: List[BBox],
    boxesR: List[BBox],
    pairs: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Для ректифицированных пар: рисуем горизонтальные линии на уровнях y каждой пары,
    чтобы визуально проверить согласование эпиполярных строк.
    """
    L = _to_bgr(imgL).copy()
    R = _to_bgr(imgR).copy()
    H_L = L.shape[0]
    H_R = R.shape[0]

    for iL, iR in pairs:
        yL = max(0, min(H_L - 1, int(round(boxesL[iL].cy))))
        yR = max(0, min(H_R - 1, int(round(boxesR[iR].cy))))
        cv.line(L, (0, yL), (L.shape[1] - 1, yL), color, thickness, cv.LINE_AA)
        cv.line(R, (0, yR), (R.shape[1] - 1, yR), color, thickness, cv.LINE_AA)

    return L, R


