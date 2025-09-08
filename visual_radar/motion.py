from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import cv2 as cv


def as_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    else:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # гарантируем uint8
    if g.dtype != np.uint8:
        g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return g


@dataclass
class DualBGModel:
    """
    Две независимые BG-модели: быстрая (для всего, включая птиц)
    и «медленная» (накопительная — дрейф облаков/тени и т.п.).
    """
    size_hw: Tuple[int, int]
    algo: str = "MOG2"
    history_fast: int = 200
    history_slow: int = 1200
    var_threshold: float = 16.0
    detect_shadows: bool = False
    learning_rate: float = -1.0   # базовый LR (auto, если < 0)
    downscale: float = 1.0        # внутренний даунскейл только для BG

    # адаптивное обучение фона
    auto_adapt: bool = True
    jump_thr: float = 12.0        # порог «глобального скачка» яркости (|Δ mean| в U8)
    freeze_len: int = 3           # на сколько кадров заморозить обучение при скачке
    night_lr: float = 0.001       # явный LR ночью, если базовый = auto
    night_scale: float = 0.3      # масштабировать явный LR ночью, если он > 0

    def __post_init__(self):
        h, w = self.size_hw
        self.size_hw = (int(h), int(w))
        # OpenCV BG
        if self.algo.upper() == "KNN":
            self.cv_fast = cv.createBackgroundSubtractorKNN(history=self.history_fast, dist2Threshold=self.var_threshold, detectShadows=self.detect_shadows)
            self.cv_slow = cv.createBackgroundSubtractorKNN(history=self.history_slow, dist2Threshold=self.var_threshold, detectShadows=self.detect_shadows)
        else:
            self.cv_fast = cv.createBackgroundSubtractorMOG2(history=self.history_fast, varThreshold=self.var_threshold, detectShadows=self.detect_shadows)
            self.cv_slow = cv.createBackgroundSubtractorMOG2(history=self.history_slow, varThreshold=self.var_threshold, detectShadows=self.detect_shadows)

        # внутреннее состояние адаптации LR
        self._ema_mean: Optional[float] = None
        self._lr_freeze: int = 0

        # Фолбэк-поля для очень старых путей (не используются при наличии cv_*):
        self.bg32: Optional[np.ndarray] = None

    # --- быстрая/медленная ветка на OpenCV
    def _resize_if_needed(self, gray: np.ndarray) -> np.ndarray:
        if float(self.downscale) != 1.0:
            return cv.resize(gray, dsize=None, fx=float(self.downscale), fy=float(self.downscale), interpolation=cv.INTER_AREA)
        return gray

    def _restore_size_mask(self, m: np.ndarray) -> np.ndarray:
        if float(self.downscale) != 1.0:
            h, w = self.size_hw
            return cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
        return m

    def _effective_lr(self, base_lr: float, night_auto: bool) -> float:
        """
        Возвращает LR с учётом ночи: если base_lr < 0 (auto), ночью берём явный маленький;
        если base_lr > 0, ночью уменьшаем его.
        """
        if not night_auto:
            return base_lr
        if base_lr < 0:
            return float(self.night_lr)
        return max(1e-6, float(base_lr) * float(self.night_scale))

    def get_adaptive_lr(self, gray_u8: np.ndarray, night_auto: bool = False) -> float:
        """
        Адаптивный learningRate:
        - при резких глобальных скачках яркости — freeze на несколько кадров (0.0)
        - ночью — замедляем обучение
        - иначе — используем заданный learning_rate (или auto)
        """
        if not self.auto_adapt:
            return self._effective_lr(float(self.learning_rate), night_auto)

        m = float(np.mean(gray_u8))
        if self._ema_mean is None:
            self._ema_mean = m

        # детект скачка яркости
        if abs(m - self._ema_mean) >= float(self.jump_thr):
            self._lr_freeze = max(self._lr_freeze, int(self.freeze_len))

        # обновим EMA тихо
        self._ema_mean = 0.9 * self._ema_mean + 0.1 * m

        if self._lr_freeze > 0:
            self._lr_freeze -= 1
            return 0.0

        return self._effective_lr(float(self.learning_rate), night_auto)

    def apply_fast(self, gray: np.ndarray, lr: float = -1.0) -> np.ndarray:
        g = as_gray(gray)
        g = self._resize_if_needed(g)
        fg = self.cv_fast.apply(g, learningRate=float(lr))
        # MOG2/KNN могут давать 0/127/255 — уберём «тени»
        _, fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)
        return self._restore_size_mask(fg)

    def apply_slow(self, gray: np.ndarray, lr: float = -1.0) -> np.ndarray:
        g = as_gray(gray)
        g = self._resize_if_needed(g)
        fg = self.cv_slow.apply(g, learningRate=float(lr))
        _, fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)
        return self._restore_size_mask(fg)

    # --- исторический питоновский фолбэк (оставлен для совместимости)
    def apply_slow_numpy(self, gray: np.ndarray, lr: float = -1.0) -> np.ndarray:
        g = as_gray(gray).astype(np.float32)
        if self.bg32 is None:
            self.bg32 = g.copy()
        alpha = 0.001 if lr < 0 else float(lr)
        cv.accumulateWeighted(g, self.bg32, alpha)
        diff = cv.absdiff(g, self.bg32).astype(np.uint8)
        _, fg = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
        return fg


def _morph_cleanup(mask: np.ndarray, size_aware: bool = True) -> np.ndarray:
    h, w = mask.shape[:2]
    k = 3
    if size_aware:
        # аккуратно масштабируем «силу» морфологии под 5Мп
        k = 3 if max(h, w) <= 1080 else 5
    k = int(max(3, min(7, k)))
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    m = cv.morphologyEx(mask, cv.MORPH_OPEN, ker, iterations=1)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, ker, iterations=1)
    return m


def find_motion_bboxes(
    gray: np.ndarray,
    bg: DualBGModel,
    min_area: int,
    max_area: int,
    thr_fast: float,
    thr_slow: float,
    use_clahe: bool = False,
    size_aware_morph: bool = True,
):
    """
    Возвращает (binary_mask, debug_vis). Боксы из маски собирает detector.py.
    """
    g = as_gray(gray)

    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    # Всегда пробуем быстрый OpenCV путь; numpy-фолбэк только если что-то сломано
    try:
        lr = float(getattr(bg, "learning_rate", -1.0))
        ms = bg.apply_fast(g, lr=lr)
    except Exception:
        ms = bg.apply_slow_numpy(g, lr=0.001)

    ms = _morph_cleanup(ms, size_aware=size_aware_morph)
    return ms, None


def make_masks_static_and_slow(
    gray: np.ndarray,
    bg: DualBGModel,
    thr_fast: float,
    thr_slow: float,
    use_clahe: bool = False,
    kernel_size: int = 3,
    night_auto: bool = False,
):
    """
    Возвращает (base_mask, slow_mask).
    base_mask - «обычное» движение (быстрое), slow_mask - накопительный дрейф (облака/тени).
    С учётом адаптивного learningRate и ночного режима.
    """
    g = as_gray(gray)
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    try:
        lr_fast = bg.get_adaptive_lr(g, night_auto=night_auto)
        lr_slow = (lr_fast * 0.2) if lr_fast > 0 else (-1.0 if lr_fast < 0 else 0.0)

        m_base = bg.apply_fast(g, lr=lr_fast)
        m_slow = bg.apply_slow(g, lr=lr_slow)
    except Exception:
        # фолбэк numpy
        m_base = bg.apply_slow_numpy(g, lr=0.005)
        m_slow = bg.apply_slow_numpy(g, lr=0.0005)

    # Лёгкая морфология: ОТКРЫТИЕ + ЗАКРЫТИЕ (зашиваем «дырочки» внутри целей)
    if kernel_size and kernel_size > 1:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
        m_base = cv.morphologyEx(m_base, cv.MORPH_OPEN, k, iterations=1)
        m_slow = cv.morphologyEx(m_slow, cv.MORPH_OPEN, k, iterations=1)
        m_base = cv.morphologyEx(m_base, cv.MORPH_CLOSE, k, iterations=1)
        m_slow = cv.morphologyEx(m_slow, cv.MORPH_CLOSE, k, iterations=1)

    return m_base, m_slow
