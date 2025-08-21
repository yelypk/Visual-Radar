
import time, json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2 as cv

def _ncc(a, b):
    a = a.astype(np.uint8); b = b.astype(np.uint8)
    if a.shape != b.shape or a.size < 36:
        return 0.0
    res = cv.matchTemplate(b, a, cv.TM_CCOEFF_NORMED)
    return float(res[0,0])

def reproject_point_Q(xL: float, y: float, xR: float, Q):
    d = float(xL - xR)
    X = Q @ np.array([xL, y, d, 1.0], dtype=np.float64)
    X /= X[3] + 1e-12
    return X[:3]  # X,Y,Z

class SnapshotSaver:
    def __init__(self, out_dir: str, min_disp: float = 1.5, min_cc: float = 0.60,
                 zmin: float = 0.0, zmax: float = 1e6, cooldown: float = 1.5,
                 pad: int = 4, debug: bool = False):
        self.out_dir = Path(out_dir)
        self.min_disp = float(min_disp)
        self.min_cc   = float(min_cc)
        self.zmin, self.zmax = float(zmin), float(zmax)
        self.cooldown = float(cooldown)
        self.pad = int(pad)
        self.debug = bool(debug)
        self._recent = []  # (t, cx, cy)

    def _too_close_recent(self, cx, cy, now):
        self._recent = [(t,x,y) for (t,x,y) in self._recent if (now - t) < self.cooldown]
        for (t,x,y) in self._recent:
            if abs(cx-x) <= 24 and abs(cy-y) <= 24:
                return True
        return False

    def maybe_save(self, rL, rR, boxL, boxR, disp: float,
                   Q=None, now: Optional[float]=None) -> bool:
        if abs(disp) < self.min_disp:
            if self.debug: print(f"[snap:skip] disp {disp:.2f} < {self.min_disp}")
            return False

        xL,yL,wL,hL = boxL; xR,yR,wR,hR = boxR
        cx, cy = xL + wL//2, yL + hL//2
        now = now or time.time()
        if self._too_close_recent(cx, cy, now):
            if self.debug: print("[snap:skip] cooldown")
            return False

        half = int(max(8, 0.5 * 3.0 * max(wL, hL, wR, hR))) + self.pad
        y1L, y2L = max(0, cy-half), min(rL.shape[0], cy+half)
        x1L, x2L = max(0, cx-half), min(rL.shape[1], cx+half)
        cR_x, cR_y = xR + wR//2, yR + hR//2
        y1R, y2R = max(0, cR_y-half), min(rR.shape[0], cR_y+half)
        x1R, x2R = max(0, cR_x-half), min(rR.shape[1], cR_x+half)
        cropL = rL[y1L:y2L, x1L:x2L]; cropR = rR[y1R:y2R, x1R:x2R]
        if cropL.size == 0 or cropR.size == 0 or cropL.shape[:2] != cropR.shape[:2]:
            if self.debug: print("[snap:skip] invalid crop sizes")
            return False
        grayL = cv.cvtColor(cropL, cv.COLOR_BGR2GRAY) if cropL.ndim==3 else cropL
        grayR = cv.cvtColor(cropR, cv.COLOR_BGR2GRAY) if cropR.ndim==3 else cropR
        cc = _ncc(grayL, grayR)
        if cc < self.min_cc:
            if self.debug: print(f"[snap:skip] NCC {cc:.2f} < {self.min_cc}")
            return False

        Z = None
        if Q is not None:
            X, Y, Z = reproject_point_Q(float(cx), float(cy), float(cR_x), Q)
            if not (self.zmin <= float(Z) <= self.zmax):
                if self.debug: print(f"[snap:skip] Z {Z:.1f} out of [{self.zmin},{self.zmax}]")
                return False

        h, w = cropL.shape[:2]
        mosaic = np.zeros((h, w*2, 3), dtype=np.uint8)
        mosaic[:, :w] = cropL
        mosaic[:,  w:] = cropR
        txt = f"d={disp:.1f}px  cc={cc:.2f}"
        if Z is not None:
            txt += f"  Z={float(Z):.1f}m"
        cv.putText(mosaic, txt, (5, max(14, h-8)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv.LINE_AA)

        ddir = self.out_dir / time.strftime("%Y%m%d"); ddir.mkdir(parents=True, exist_ok=True)
        base = time.strftime("%H%M%S") + f"_{int((now-int(now))*1000):03d}"
        png  = ddir / f"{base}.png"
        cv.imwrite(str(png), mosaic)

        meta = {
            "timestamp": now,
            "left_box":  [int(v) for v in (xL,yL,wL,hL)],
            "right_box": [int(v) for v in (xR,yR,wR,hR)],
            "disp": float(disp),
            "Z": None if Z is None else float(Z),
            "crop_shape": [int(h), int(w)]
        }
        with open(ddir / f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self._recent.append((now, cx, cy))
        if self.debug: print("[snap] saved", png)
        return True
