from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional
import numpy as np
from math import hypot

from visual_radar.utils import BBox


def _iou_xywh(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


@dataclass
class Trk:
    """Внутреннее состояние одного трека."""
    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float
    id: int
    age: int = 0
    misses: int = 0
    last_cx: Optional[float] = None
    last_cy: Optional[float] = None
    move_hist: List[float] = field(default_factory=list)

    def push(self, cx: float, cy: float, win: int = 8) -> None:
        move = 0.0 if self.last_cx is None else hypot(cx - self.last_cx, cy - self.last_cy)
        self.move_hist.append(move)
        if len(self.move_hist) > win:
            self.move_hist.pop(0)
        self.last_cx, self.last_cy = cx, cy

    def total_move(self) -> float:
        return float(sum(self.move_hist))

    def avg_speed(self) -> float:
        n = max(1, len(self.move_hist))
        return float(sum(self.move_hist) / n)


class BoxTracker:
    """
    Простой IoU-трекер с гистерезисом.
    Бокс показывается, если трек НЕ пропущен на кадре и:
      age >= min_age  И
      (total_move >= min_disp ИЛИ avg_speed >= min_speed)
    """
    def __init__(self,
                 iou_thr: float = 0.3,
                 min_age: int = 3,
                 max_missed: int = 3,
                 min_disp: float = 4.0,
                 min_speed: float = 1.0):
        self.iou_thr = float(iou_thr)
        self.min_age = int(min_age)
        self.max_missed = int(max_missed)
        self.min_disp = float(min_disp)
        self.min_speed = float(min_speed)
        self.tracks: List[Trk] = []
        self.next_id: int = 1

    def _match(self, det_xywh: List[Tuple[float, float, float, float]]) -> List[Tuple[int, int]]:
        """Грубое сопоставление по максимальному IoU (жадно)."""
        M = len(self.tracks); N = len(det_xywh)
        if M == 0 or N == 0:
            return []

        iou = np.zeros((M, N), dtype=np.float32)
        for ti, t in enumerate(self.tracks):
            tb = (t.x, t.y, t.w, t.h)
            for di, db in enumerate(det_xywh):
                iou[ti, di] = _iou_xywh(tb, db)

        matches: List[Tuple[int, int]] = []
        used_t: Set[int] = set()
        used_d: Set[int] = set()
        while True:
            ti, di = np.unravel_index(iou.argmax(), iou.shape)
            score = float(iou[ti, di])
            if score < self.iou_thr:
                break
            if ti in used_t or di in used_d:
                iou[ti, di] = -1.0
                continue
            matches.append((ti, di))
            used_t.add(ti); used_d.add(di)
            # блокируем строку/столбец
            iou[ti, :] = -1.0
            iou[:, di] = -1.0
        return matches

    def update(self, boxes: List[BBox]) -> Set[int]:
        """
        Обновить трекер по списку детекций.
        Возвращает множество ИНДЕКСОВ детекций (из входного списка),
        которые прошли критерии показа.
        """
        det_xywh = [(float(b.x), float(b.y), float(b.w), float(b.h)) for b in boxes]
        det_cent = [(float(b.cx), float(b.cy)) for b in boxes]
        N = len(boxes)

        # 1) сопоставляем текущие треки и детекции
        matches = self._match(det_xywh)
        matched_d: Set[int] = {di for _, di in matches}
        matched_t: Set[int] = {ti for ti, _ in matches}

        # 2) обновляем совпавшие треки
        for ti, di in matches:
            x, y, w, h = det_xywh[di]
            cx, cy = det_cent[di]
            t = self.tracks[ti]
            t.x, t.y, t.w, t.h = x, y, w, h
            t.cx, t.cy = cx, cy
            t.age += 1
            t.misses = 0
            t.push(cx, cy)

        # 3) создаём новые треки из неиспользованных детекций
        for di in range(N):
            if di in matched_d:
                continue
            x, y, w, h = det_xywh[di]
            cx, cy = det_cent[di]
            t = Trk(x=x, y=y, w=w, h=h, cx=cx, cy=cy, id=self.next_id, age=1, misses=0)
            t.push(cx, cy)
            self.tracks.append(t)
            self.next_id += 1

        # 4) увеличиваем счётчик пропусков; удаляем «просроченные»
        keep_tracks: List[Trk] = []
        for ti, t in enumerate(self.tracks):
            if ti in matched_t:
                keep_tracks.append(t)
            else:
                t.misses += 1
                if t.misses <= self.max_missed:
                    keep_tracks.append(t)
        self.tracks = keep_tracks

        # 5) критерии показа — возвращаем индексы детекций, которые стоит рисовать
        to_show: Set[int] = set()
        det_to_trk: dict[int, Trk] = {}

        if len(self.tracks) and N:
            # пересобираем «быструю» связь детекция→трек по лучшему IoU
            for ti, t in enumerate(self.tracks):
                best_iou, best_di = 0.0, -1
                tb = (t.x, t.y, t.w, t.h)
                for di, db in enumerate(det_xywh):
                    i = _iou_xywh(tb, db)
                    if i > best_iou:
                        best_iou, best_di = i, di
                if best_di >= 0:
                    det_to_trk[best_di] = t

        for di in range(N):
            t = det_to_trk.get(di)
            if not t or t.misses > 0 or t.age < self.min_age:
                continue
            if (t.total_move() >= self.min_disp) or (t.avg_speed() >= self.min_speed):
                to_show.add(di)

        return to_show

