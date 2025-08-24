"""
Snapshot saving utilities for Visual Radar.
"""

import os
import time
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

class SnapshotSaver:
    """
    Saves snapshots of detected objects from stereo frames.
    """
    def __init__(
        self,
        out_dir: str,
        min_disp: float = 1.5,
        min_cc: float = 0.6,
        cooldown: float = 1.5,
        pad: int = 4,
        debug: bool = False,
    ):
        self.out_dir = out_dir
        self.min_disp = min_disp
        self.min_cc = min_cc
        self.cooldown = cooldown
        self.pad = pad
        self.debug = debug
        self.last_save_time = 0.0

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

    def maybe_save(
        self,
        rectL: np.ndarray,
        rectR: np.ndarray,
        boxL: Tuple[int, int, int, int],
        boxR: Tuple[int, int, int, int],
        disp: float,
        Q: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Save snapshot if conditions are met (cooldown, min_disp, etc.).
        Returns the saved file path or None.
        """
        now = time.time()
        if (now - self.last_save_time) < self.cooldown:
            if self.debug:
                print("[SnapshotSaver] Cooldown active, skipping save.")
            return None
        if abs(disp) < self.min_disp:
            if self.debug:
                print(f"[SnapshotSaver] Disp {disp:.2f} below min_disp {self.min_disp}.")
            return None

        # Crop and pad boxes
        xL, yL, wL, hL = boxL
        xR, yR, wR, hR = boxR
        pad = self.pad

        cropL = rectL[max(0, yL-pad):yL+hL+pad, max(0, xL-pad):xL+wL+pad]
        cropR = rectR[max(0, yR-pad):yR+hR+pad, max(0, xR-pad):xR+wR+pad]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"snap_{timestamp}_{int(disp)}.png"
        path = os.path.join(self.out_dir, fname)

        # Stack crops side by side
        snap = np.hstack([cropL, cropR])
        cv.imwrite(path, snap)
        self.last_save_time = now

        if self.debug:
            print(f"[SnapshotSaver] Saved snapshot: {path}")
        return path