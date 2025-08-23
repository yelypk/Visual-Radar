 # visual_radar/snapshots.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2 as cv
import numpy as np

from visual_radar.utils import now_s


@dataclass
class SnapshotSaver:
    out_dir: str = "detections"
    min_disp: float = 1.5     # минимальный параллакс для сохранения
    min_cc: float = 0.6       # минимальная "похожесть" L/R-кропов (норм. кросс-корреляция)
    cooldown: float = 1.5     # сек между сохранениями
    pad: int = 4              # паддинг вокруг боксов
    debug: bool = False

    _last_ts: float = 0.0

    def _ensure_dir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _clip_box(xywh: Tuple[int, int, int, int], w: int, h: int, pad: int) -> Tuple[int, int, int, int]:
        x, y, bw, bh = xywh
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2 * pad)
        bh = min(h - y, bh + 2 * pad)
        return int(x), int(y), int(max(1, bw)), int(max(1, bh))

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        """Быстрая норм. кросс-корреляция на одномерных векторах."""
        if a.size == 0 or b.size == 0:
            return 0.0
        av = a.astype(np.float32).ravel()
        bv = b.astype(np.float32).ravel()
        av -= av.mean()
        bv -= bv.mean()
        sa = float(np.sqrt((av * av).sum()) + 1e-6)
        sb = float(np.sqrt((bv * bv).sum()) + 1e-6)
        return float((av @ bv) / (sa * sb))

    def maybe_save(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
        boxL_xywh: Tuple[int, int, int, int],
        boxR_xywh: Tuple[int, int, int, int],
        disp: float,
        Q: Optional[np.ndarray] = None,
    ) -> None:
        """Сохранить кропы L/R, если условия выполняются."""
        t = now_s()
        if (t - self._last_ts) < self.cooldown:
            return
        if abs(float(disp)) < float(self.min_disp):
            return

        h, w = imgL.shape[:2]
        xL, yL, wL, hL = self._clip_box(boxL_xywh, w, h, self.pad)
        xR, yR, wR, hR = self._clip_box(boxR_xywh, w, h, self.pad)

        cropL = imgL[yL : yL + hL, xL : xL + wL]
        cropR = imgR[yR : yR + hR, xR : xR + wR]
        if cropL.size == 0 or cropR.size == 0:
            return

        # нормализуем кропы к одному размеру для оценки схожести
        tgt_w = max(16, min(wL, wR))
        tgt_h = max(16, min(hL, hR))
        gL = cv.cvtColor(cv.resize(cropL, (tgt_w, tgt_h), interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)
        gR = cv.cvtColor(cv.resize(cropR, (tgt_w, tgt_h), interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)
        cc = self._ncc(gL, gR)
        if cc < float(self.min_cc):
            if self.debug:
                print(f"[snapshots] drop by cc={cc:.2f} < {self.min_cc}")
            return

        self._ensure_dir()
        ts = int(t * 1000)
        base = f"det_{ts}_disp{abs(disp):.2f}_cc{cc:.2f}"
        pathL = os.path.join(self.out_dir, base + "_L.jpg")
        pathR = os.path.join(self.out_dir, base + "_R.jpg")

        cv.imwrite(pathL, cropL)
        cv.imwrite(pathR, cropR)

        if self.debug:
            print(f"[snapshots] saved: {os.path.basename(pathL)}, {os.path.basename(pathR)}")

        self._last_ts = t

