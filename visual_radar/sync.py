# visual_radar/sync.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple


@dataclass
class Frame:
    """Простая обёртка кадра с меткой времени (секунды, float)."""
    img: "object"  # np.ndarray, но специально без жёсткой зависимости
    ts: float


def pair_nearest(
    seqL: Iterable[Frame],
    seqR: Iterable[Frame],
    max_dt: float = 0.050,
) -> Iterator[Tuple[Frame, Frame]]:
    """
    Сопоставляет два потока кадров по ближайшим меткам времени.
    Если |tsL - tsR| <= max_dt → выдаёт пару (L, R).
    Иначе продвигаем «отстающий» поток.

    Пример использования:
        for fL, fR in pair_nearest(buffL, buffR, max_dt=0.03):
            process(fL.img, fR.img)
    """
    itL = iter(seqL)
    itR = iter(seqR)
    try:
        fL = next(itL)
        fR = next(itR)
    except StopIteration:
        return

    while True:
        dt = float(fL.ts - fR.ts)
        if abs(dt) <= float(max_dt):
            yield fL, fR
            try:
                fL = next(itL)
                fR = next(itR)
            except StopIteration:
                return
        elif dt > 0.0:
            # левый «вперёд» → подтягиваем правый
            try:
                fR = next(itR)
            except StopIteration:
                return
        else:
            # правый «вперёд» → подтягиваем левый
            try:
                fL = next(itL)
            except StopIteration:
                return
