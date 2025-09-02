from __future__ import annotations

import time
from dataclasses import dataclass

def monotonic_s() -> float:
    return float(time.monotonic())

def now_s() -> float:
    return monotonic_s()

def wallclock_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


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

    def to_int(self) -> tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)
