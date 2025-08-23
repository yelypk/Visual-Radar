from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import threading
import queue
import time

import cv2 as cv
import numpy as np

from visual_radar.utils import wallclock_stamp, BBox


@dataclass
class _Job:
    path: Path
    img: np.ndarray
    quality: int


class SnapshotSaver:
    """
    Лёгкие снапшоты:
      - JPEG качество настраивается,
      - rate-limit (кадров/сек),
      - фоновая запись, чтобы не стопорить пайплайн.
    API совместим с вашими вызовами: maybe_save(L, R, bboxL, bboxR, disp, Q=None)
    """

    def __init__(
        self,
        out_dir: str = "detections",
        min_disp: float = 1.5,
        min_cc: float = 0.6,
        cooldown: float = 1.5,
        pad: int = 4,
        debug: bool = False,
        jpeg_quality: int = 92,
        max_rate: float = 2.0,      # не чаще N кадров/сек
        max_queue: int = 4,         # очередь фоновой записи
    ):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_disp = float(min_disp)
        self.min_cc = float(min_cc)
        self.cooldown = float(cooldown)
        self.pad = int(pad)
        self.debug = bool(debug)
        self.jpeg_quality = int(np.clip(jpeg_quality, 60, 100))
        self.max_rate = float(max_rate)
        self._min_dt = 1.0 / max(1e-6, self.max_rate)

        self._last_ts = 0.0
        self._q: "queue.Queue[_Job]" = queue.Queue(maxsize=int(max_queue))
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                cv.imwrite(str(job.path), job.img, [cv.IMWRITE_JPEG_QUALITY, int(job.quality)])
            except Exception:
                pass

    def _schedule(self, path: Path, img: np.ndarray):
        try:
            self._q.put_nowait(_Job(path, img, self.jpeg_quality))
        except queue.Full:
            # очередь заполнена — пропускаем, чтобы не тормозить пайплайн
            pass

    def maybe_save(
        self,
        rectL: np.ndarray,
        rectR: np.ndarray,
        boxL: Tuple[int, int, int, int],
        boxR: Tuple[int, int, int, int],
        disp: float,
        Q=None,
    ) -> None:
        # проста эвристика — требуем минимальный диспаратет
        if abs(float(disp)) < self.min_disp:
            return

        t = time.monotonic()
        if (t - self._last_ts) < max(self.cooldown, self._min_dt):
            return
        self._last_ts = t

        xL, yL, wL, hL = boxL
        xR, yR, wR, hR = boxR
        h, w = rectL.shape[:2]

        # подрезаем с паддингом и границами
        def _crop(img, x, y, w, h, pad):
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad); y2 = min(img.shape[0], y + h + pad)
            return img[y1:y2, x1:x2]

        cutL = _crop(rectL, xL, yL, wL, hL, self.pad)
        cutR = _crop(rectR, xR, yR, wR, hR, self.pad)

        # собираем превью L|R
        H = max(cutL.shape[0], cutR.shape[0])
        def _fit(himg):
            if himg.shape[0] != H:
                s = H / float(himg.shape[0])
                himg = cv.resize(himg, (int(round(himg.shape[1] * s)), H), interpolation=cv.INTER_AREA)
            return himg
        cut = np.hstack([_fit(cutL), _fit(cutR)])

        name = f"{wallclock_stamp()}_disp{abs(disp):.1f}.jpg"
        self._schedule(self.dir / name, cut)

    def close(self):
        self._stop.set()
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass

