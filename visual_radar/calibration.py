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

    # Фолбэк: проективная (обычно — идентичность)
    H1: Optional[np.ndarray] = None
    H2: Optional[np.ndarray] = None

    # Метрическая ректификация — float-карты (универсальные; можно ресайзить)
    map1x: Optional[np.ndarray] = None  # CV_32FC1
    map1y: Optional[np.ndarray] = None  # CV_32FC1
    map2x: Optional[np.ndarray] = None  # CV_32FC1
    map2y: Optional[np.ndarray] = None  # CV_32FC1

    # Метрическая ректификация — быстрые fixed-point карты для remap (предпочтительны при точном размере)
    map1L_s16: Optional[np.ndarray] = None  # CV_16SC2
    map2L_u16: Optional[np.ndarray] = None  # CV_16UC1
    map1R_s16: Optional[np.ndarray] = None  # CV_16SC2
    map2R_u16: Optional[np.ndarray] = None  # CV_16UC1

    # Матрица перекидывания диспаритета в 3D (если есть)
    Q: Optional[np.ndarray] = None

    # кэш для быстрого ресайза float-карт
    _cache_size: Optional[Tuple[int, int]] = None
    _cache_maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None


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
    """
    Готовим и float-, и fixed-point (CV_16SC2/CV_16UC1) карты.
    По докам OpenCV, remap с CV_16SC2 быстрее; float-карты держим для случая,
    когда фактический размер кадра отличается (их можно качественно ресайзить).
    """
    w, h = frame_size

    # Поле зрения без кропа
    newK1, _ = cv.getOptimalNewCameraMatrix(K1, D1, (w, h), alpha)
    newK2, _ = cv.getOptimalNewCameraMatrix(K2, D2, (w, h), alpha)

    R1, R2, P1, P2, Q, _roi1, _roi2 = cv.stereoRectify(
        newK1, D1, newK2, D2, (w, h), R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=alpha
    )

    # 1) быстрые карты: CV_16SC2/CV_16UC1
    m1L_s16, m2L_u16 = cv.initUndistortRectifyMap(newK1, D1, R1, P1, (w, h), cv.CV_16SC2)
    m1R_s16, m2R_u16 = cv.initUndistortRectifyMap(newK2, D2, R2, P2, (w, h), cv.CV_16SC2)

    # 2) float-карты: CV_32FC1 (их можно ресайзить)
    m1x, m1y = cv.initUndistortRectifyMap(newK1, D1, R1, P1, (w, h), cv.CV_32FC1)
    m2x, m2y = cv.initUndistortRectifyMap(newK2, D2, R2, P2, (w, h), cv.CV_32FC1)

    return Calibration(
        mode="metric_maps",
        size_ref=(w, h),
        map1x=m1x, map1y=m1y, map2x=m2x, map2y=m2y,
        map1L_s16=m1L_s16, map2L_u16=m2L_u16,
        map1R_s16=m1R_s16, map2R_u16=m2R_u16,
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
    """
    Применить ректификацию/ремап без кропа (full FOV).
    Используем быстрые fixed-point карты, если размер кадра совпадает с эталонным;
    если нет — аккуратно ресайзим float-карты под текущий размер.
    """
    if calib.mode == "metric_maps" and (calib.map1x is not None or calib.map1L_s16 is not None):
        h, w = imgL.shape[:2]
        mw, mh = calib.size_ref

        if (w, h) == (mw, mh) and calib.map1L_s16 is not None:
            # Быстрый путь: CV_16SC2/CV_16UC1
            rL = cv.remap(imgL, calib.map1L_s16, calib.map2L_u16,
                          interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
            rR = cv.remap(imgR, calib.map1R_s16, calib.map2R_u16,
                          interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
            return rL, rR

        # Медленный, но универсальный путь: float-карты, с кэшем под целевой размер
        if calib._cache_size != (w, h):
            map1x = cv.resize(calib.map1x, (w, h), interpolation=cv.INTER_LINEAR)
            map1y = cv.resize(calib.map1y, (w, h), interpolation=cv.INTER_LINEAR)
            map2x = cv.resize(calib.map2x, (w, h), interpolation=cv.INTER_LINEAR)
            map2y = cv.resize(calib.map2y, (w, h), interpolation=cv.INTER_LINEAR)
            calib._cache_size = (w, h)
            calib._cache_maps = (map1x, map1y, map2x, map2y)
            if calib.Q is not None and (w, h) != (mw, mh):
                # для точной глубины лучше пересчитать калибровку под новый размер
                print("[!] Q is for size", (mw, mh), "— recompute calibration for accurate depth at", (w, h))

        map1x, map1y, map2x, map2y = calib._cache_maps
        rL = cv.remap(imgL, map1x, map1y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        rR = cv.remap(imgR, map2x, map2y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        return rL, rR

    # Фолбэк: проективно (по умолчанию — идентичность)
    H1 = calib.H1 if calib.H1 is not None else np.eye(3, dtype=np.float64)
    H2 = calib.H2 if calib.H2 is not None else np.eye(3, dtype=np.float64)
    w = imgL.shape[1]; h = imgL.shape[0]
    rL = cv.warpPerspective(imgL, H1, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    rR = cv.warpPerspective(imgR, H2, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return rL, rR
