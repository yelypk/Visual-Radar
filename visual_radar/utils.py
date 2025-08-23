from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import time


def now_s() -> float:
    """Epoch time в секундах (float). Отдельная функция, чтобы не тянуть зависимости."""
    return time.time()


def clamp(v: float, lo: float, hi: float) -> float:
    """Ограничить значение в [lo, hi]."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return float(self.x + self.w * 0.5)

    @property
    def cy(self) -> float:
        return float(self.y + self.h * 0.5)

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return float(self.x), float(self.y), float(self.w), float(self.h)

    def area(self) -> float:
        return max(0.0, float(self.w)) * max(0.0, float(self.h))

    def scaled(self, sx: float, sy: float) -> "BBox":
        """Вернуть масштабированную копию (без мутации исходного бокса)."""
        return BBox(self.x * sx, self.y * sy, self.w * sx, self.h * sy)

    def to_int(self) -> Tuple[int, int, int, int]:
        """Целочисленные координаты (для cv2)."""
        return int(self.x), int(self.y), int(self.w), int(self.h)
