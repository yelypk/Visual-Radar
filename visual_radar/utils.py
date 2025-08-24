from __future__ import annotations

import time
from dataclasses import dataclass

# --- TIME UTILITIES ---

def monotonic_s() -> float:
    """
    Monotonic seconds (not affected by system clock changes).
    """
    return float(time.monotonic())

def now_s() -> float:
    """
    Alias for monotonic_s, for compatibility with existing imports.
    """
    return monotonic_s()

def wallclock_stamp() -> str:
    """
    Readable local time stamp for file names.
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

# --- GEOMETRY ---

@dataclass
class BBox:
    """
    Bounding box in (x, y, w, h) format.
    """
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        """
        Center x-coordinate.
        """
        return float(self.x + self.w * 0.5)

    @property
    def cy(self) -> float:
        """
        Center y-coordinate.
        """
        return float(self.y + self.h * 0.5)

    def to_int(self) -> tuple[int, int, int, int]:
        """
        Convert bounding box coordinates to integers.
        """
        return int(self.x), int(self.y), int(self.w), int(self.h)
