from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import numpy as np

def _iou_xywh(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


@dataclass
class Trk:
    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float
    id: int
    age: int = 0        
    misses: int = 0      

class BoxTracker:
    def __init__(self, iou_thr: float = 0.3, min_age: int = 3, max_missed: int = 3):
        self.iou_thr = float(iou_thr)
        self.min_age = int(min_age)
        self.max_missed = int(max_missed)
        self.tracks: List[Trk] = []
        self.next_id: int = 1

    def update(self, boxes: Optional[List]) -> Set[int]:
        if boxes is None:
            boxes = []

        det: List[Tuple[float, float, float, float, float, float]] = []
        for b in boxes:
            det.append(
                (float(b.x), float(b.y), float(b.w), float(b.h),
                 float(b.cx), float(b.cy))
            )

        N = len(det)
        M = len(self.tracks)

        # посчитаем IoU матрицу (trk x det)
        if M > 0 and N > 0:
            iou_mat = np.zeros((M, N), dtype=np.float32)
            for ti, t in enumerate(self.tracks):
                tb = (t.x, t.y, t.w, t.h)
                for di, (x, y, w, h, _, _) in enumerate(det):
                    iou_mat[ti, di] = _iou_xywh(tb, (x, y, w, h))
        else:
            iou_mat = None

        # гриди-матчинг по IoU
        matches: List[Tuple[int, int, float]] = []
        used_trk: Set[int] = set()
        used_det: Set[int] = set()
        if iou_mat is not None:
            while True:
                ti, di = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                score = float(iou_mat[ti, di])
                if score < self.iou_thr:
                    break
                if ti in used_trk or di in used_det:
                    iou_mat[ti, di] = -1.0
                    continue
                matches.append((ti, di, score))
                used_trk.add(ti)
                used_det.add(di)
                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

        matched_trk = {ti for ti, _, _ in matches}
        matched_det = {di for _, di, _ in matches}

        for ti, di, _ in matches:
            x, y, w, h, cx, cy = det[di]
            tr = self.tracks[ti]
            tr.x, tr.y, tr.w, tr.h = x, y, w, h
            tr.cx, tr.cy = cx, cy
            tr.age += 1
            tr.misses = 0

        for ti in range(M):
            if ti not in matched_trk:
                self.tracks[ti].misses += 1

        for di in range(N):
            if di not in matched_det:
                x, y, w, h, cx, cy = det[di]
                self.tracks.append(Trk(x, y, w, h, cx, cy, id=self.next_id))
                self.next_id += 1

        stable_det_idx: Set[int] = set()
        det_to_trk = {di: ti for ti, di, _ in matches}
        for di, ti in det_to_trk.items():
            trk = self.tracks[ti]
            if trk.misses == 0 and trk.age >= self.min_age:
                stable_det_idx.add(di)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_missed]

        return stable_det_idx
