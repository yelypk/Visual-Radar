# visual_radar/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2 as cv


@dataclass
class Calibration:
    """Ректификация для заданного размера кадра."""
    mode: str  # "proj" | "metric_maps"
    size_ref: Tuple[int, int]  # (w, h)
    # Проективная (фолбэк) ректификация
    H1: Optional[np.ndarray] = None
    H2: Optional[np.ndarray] = None
    # Метрическая ректификация
    map1x: Optional[np.ndarray] = None
    map1y: Optional[np.ndarray] = None
    map2x: Optional[np.ndarray] = None
    map2y: Optional[np.ndarray] = None
    # Матрица перекидывания диспаритета в 3D (если есть)
    Q: Optional[np.ndarray] = None


def _try_load_npz(npz_path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if npz_path.exists():
        with np.load(str(npz_path)) as f:
            for k in f.files:
                data[k] = f[k]
    return data


def _metric_from_intrinsics(
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    frame_size: Tuple[int, int],
    alpha: float = 1.0,
) -> Calibration:
    w, h = frame_size
    # Полный угол без кропа
    newK1, _ = cv.getOptimalNewCameraMatrix(K1, D1, (w, h), alpha)
    newK2, _ = cv.getOptimalNewCameraMatrix(K2, D2, (w, h), alpha)

    R1, R2, P1, P2, Q, _roi1, _roi2 = cv.stereoRectify(
        newK1, D1, newK2, D2, (w, h), R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=alpha
    )

    map1x, map1y = cv.initUndistortRectifyMap(newK1, D1, R1, P1, (w, h), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(newK2, D2, R2, P2, (w, h), cv.CV_32FC1)

    return Calibration(
        mode="metric_maps",
        size_ref=(w, h),
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
        Q=Q
    )


def _proj_identity(frame_size: Tuple[int, int]) -> Calibration:
    w, h = frame_size
    H1 = np.eye(3, dtype=np.float64)
    H2 = np.eye(3, dtype=np.float64)
    return Calibration(mode="proj", size_ref=(w, h), H1=H1, H2=H2, Q=None)


def load_calibration(
    calib_dir: Path,
    intrinsics: Optional[Path],
    frame_size: Tuple[int, int],
    baseline_m: Optional[float] = None,  # оставлено для совместимости сигнатур
) -> Calibration:
    """
    Пытаемся загрузить и/или построить ректификацию для (width,height).

    Приоритет:
    1) Указанный intrinsics .npz с K1,K2,D1,D2,R,T → считаем карты.
    2) <calib_dir>/intrinsics.npz → считаем карты.
    3) Фолбэк: проективная идентичность (пропуск без изменений).
    """
    w, h = frame_size

    # 1) Явный путь
    if intrinsics is not None:
        data = _try_load_npz(Path(intrinsics))
        K1 = data.get("K1"); K2 = data.get("K2")
        D1 = data.get("D1"); D2 = data.get("D2")
        R  = data.get("R");  T  = data.get("T")
        if all(x is not None for x in (K1, D1, K2, D2, R, T)):
            return _metric_from_intrinsics(K1, D1, K2, D2, R, T, (w, h), alpha=1.0)

    # 2) Директория калибровки
    npz_path = Path(calib_dir) / "intrinsics.npz"
    data = _try_load_npz(npz_path)
    if data:
        K1 = data.get("K1"); K2 = data.get("K2")
        D1 = data.get("D1"); D2 = data.get("D2")
        R  = data.get("R");  T  = data.get("T")
        if all(x is not None for x in (K1, D1, K2, D2, R, T)):
            return _metric_from_intrinsics(K1, D1, K2, D2, R, T, (w, h), alpha=1.0)

    # 3) Фолбэк
    return _proj_identity((w, h))


def rectified_pair(calib: Calibration, imgL, imgR):
    """Применить ректификацию/ремап без кропа (full FOV)."""
    if calib.mode == "metric_maps" and calib.map1x is not None:
        # Если размер кадра поменялся — приблизительно рескейлим карты (для точной 3D лучше пересчитать).
        h, w = imgL.shape[:2]
        mw, mh = calib.size_ref
        if (w, h) != (mw, mh):
            map1x = cv.resize(calib.map1x, (w, h), interpolation=cv.INTER_LINEAR)
            map1y = cv.resize(calib.map1y, (w, h), interpolation=cv.INTER_LINEAR)
            map2x = cv.resize(calib.map2x, (w, h), interpolation=cv.INTER_LINEAR)
            map2y = cv.resize(calib.map2y, (w, h), interpolation=cv.INTER_LINEAR)
            if calib.Q is not None:
                print("[!] Q is for size", (mw, mh), "— recompute calibration for accurate depth at", (w, h))
        else:
            map1x, map1y, map2x, map2y = calib.map1x, calib.map1y, calib.map2x, calib.map2y

        rL = cv.remap(imgL, map1x, map1y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        rR = cv.remap(imgR, map2x, map2y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        return rL, rR

    # Фолбэк: проективно (по умолчанию — идентичность)
    H1 = calib.H1 if calib.H1 is not None else np.eye(3, dtype=np.float64)
    H2 = calib.H2 if calib.H2 is not None else np.eye(3, dtype=np.float64)
    w = imgL.shape[1]; h = imgL.shape[0]
    rL = cv.warpPerspective(imgL, H1, (w, h))
    rR = cv.warpPerspective(imgR, H2, (w, h))
    return rL, rR
