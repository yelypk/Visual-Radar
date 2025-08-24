from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2 as cv
import logging


@dataclass
class Calibration:
    """
    Rectification for a given frame size.
    """
    mode: str  # "proj" | "metric_maps"
    size_ref: Tuple[int, int]  # (width, height)

    # Fallback: projective (usually identity)
    H1: Optional[np.ndarray] = None
    H2: Optional[np.ndarray] = None

    # Metric rectification — float maps (universal; can be resized)
    map1x: Optional[np.ndarray] = None  # CV_32FC1
    map1y: Optional[np.ndarray] = None  # CV_32FC1
    map2x: Optional[np.ndarray] = None  # CV_32FC1
    map2y: Optional[np.ndarray] = None  # CV_32FC1

    # Metric rectification — fast fixed-point maps for remap (preferred for exact size)
    map1L_s16: Optional[np.ndarray] = None  # CV_16SC2
    map2L_u16: Optional[np.ndarray] = None  # CV_16UC1
    map1R_s16: Optional[np.ndarray] = None  # CV_16SC2
    map2R_u16: Optional[np.ndarray] = None  # CV_16UC1

    # Matrix for disparity to 3D conversion (if available)
    Q: Optional[np.ndarray] = None

    # Cache for fast resizing of float maps
    _cache_size: Optional[Tuple[int, int]] = None
    _cache_maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None


def _try_load_npz(npz_path: Path) -> Dict[str, Any]:
    """
    Attempt to load a .npz file and return its contents as a dictionary.
    """
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
    Prepare both float and fixed-point (CV_16SC2/CV_16UC1) maps.
    According to OpenCV docs, remap with CV_16SC2 is faster; float maps are kept for resizing to other frame sizes.
    """
    width, height = frame_size

    # Field of view without cropping
    newK1, _ = cv.getOptimalNewCameraMatrix(K1, D1, (width, height), alpha)
    newK2, _ = cv.getOptimalNewCameraMatrix(K2, D2, (width, height), alpha)

    R1, R2, P1, P2, Q, _roi1, _roi2 = cv.stereoRectify(
        newK1, D1, newK2, D2, (width, height), R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=alpha
    )

    # 1) Fast maps: CV_16SC2/CV_16UC1
    map1L_s16, map2L_u16 = cv.initUndistortRectifyMap(newK1, D1, R1, P1, (width, height), cv.CV_16SC2)
    map1R_s16, map2R_u16 = cv.initUndistortRectifyMap(newK2, D2, R2, P2, (width, height), cv.CV_16SC2)

    # 2) Float maps: CV_32FC1 (can be resized)
    map1x, map1y = cv.initUndistortRectifyMap(newK1, D1, R1, P1, (width, height), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(newK2, D2, R2, P2, (width, height), cv.CV_32FC1)

    return Calibration(
        mode="metric_maps",
        size_ref=(width, height),
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        map1L_s16=map1L_s16, map2L_u16=map2L_u16,
        map1R_s16=map1R_s16, map2R_u16=map2R_u16,
        Q=Q
    )


def _proj_identity(frame_size: Tuple[int, int]) -> Calibration:
    """
    Return a projective identity calibration (pass-through).
    """
    width, height = frame_size
    H1 = np.eye(3, dtype=np.float64)
    H2 = np.eye(3, dtype=np.float64)
    return Calibration(mode="proj", size_ref=(width, height), H1=H1, H2=H2, Q=None)


def _load_intrinsics(data: Dict[str, Any], frame_size: Tuple[int, int]) -> Optional[Calibration]:
    """
    Load calibration from intrinsics dictionary if all required keys are present.
    """
    K1 = data.get("K1")
    K2 = data.get("K2")
    D1 = data.get("D1")
    D2 = data.get("D2")
    R = data.get("R")
    T = data.get("T")
    if all(x is not None for x in (K1, D1, K2, D2, R, T)):
        return _metric_from_intrinsics(K1, D1, K2, D2, R, T, frame_size, alpha=1.0)
    return None


def load_calibration(
    calib_dir: Path,
    intrinsics: Optional[Path],
    frame_size: Tuple[int, int],
    baseline_m: Optional[float] = None,
) -> Calibration:
    """
    Try to load and/or build rectification for (width, height).

    Priority:
    1) Provided intrinsics .npz with K1, K2, D1, D2, R, T → compute maps.
    2) <calib_dir>/intrinsics.npz → compute maps.
    3) Fallback: projective identity (pass-through).
    """
    # 1) Explicit path
    if intrinsics is not None:
        data = _try_load_npz(Path(intrinsics))
        calib = _load_intrinsics(data, frame_size)
        if calib:
            return calib

    # 2) Calibration directory
    npz_path = Path(calib_dir) / "intrinsics.npz"
    data = _try_load_npz(npz_path)
    calib = _load_intrinsics(data, frame_size)
    if calib:
        return calib

    # 3) Fallback
    return _proj_identity(frame_size)


def rectified_pair(
    calib: Calibration,
    img_left: np.ndarray,
    img_right: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rectification/remap without cropping (full FOV).
    Uses fast fixed-point maps if frame size matches reference;
    otherwise, resizes float maps for current size.
    """
    if calib.mode == "metric_maps" and (calib.map1x is not None or calib.map1L_s16 is not None):
        height, width = img_left.shape[:2]
        ref_width, ref_height = calib.size_ref

        if (width, height) == (ref_width, ref_height) and calib.map1L_s16 is not None:
            # Fast path: CV_16SC2/CV_16UC1
            rectified_left = cv.remap(
                img_left, calib.map1L_s16, calib.map2L_u16,
                interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )
            rectified_right = cv.remap(
                img_right, calib.map1R_s16, calib.map2R_u16,
                interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )
            return rectified_left, rectified_right

        # Slow but universal path: float maps, cached for target size
        if calib._cache_size != (width, height):
            map1x = cv.resize(calib.map1x, (width, height), interpolation=cv.INTER_LINEAR)
            map1y = cv.resize(calib.map1y, (width, height), interpolation=cv.INTER_LINEAR)
            map2x = cv.resize(calib.map2x, (width, height), interpolation=cv.INTER_LINEAR)
            map2y = cv.resize(calib.map2y, (width, height), interpolation=cv.INTER_LINEAR)
            calib._cache_size = (width, height)
            calib._cache_maps = (map1x, map1y, map2x, map2y)
            if calib.Q is not None and (width, height) != (ref_width, ref_height):
                logging.warning(
                    "Q matrix is for size %s — recompute calibration for accurate depth at %s",
                    (ref_width, ref_height), (width, height)
                )

        map1x, map1y, map2x, map2y = calib._cache_maps
        rectified_left = cv.remap(
            img_left, map1x, map1y,
            interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
        )
        rectified_right = cv.remap(
            img_right, map2x, map2y,
            interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
        )
        return rectified_left, rectified_right

    # Fallback: projective (default — identity)
    H1 = calib.H1 if calib.H1 is not None else np.eye(3, dtype=np.float64)
    H2 = calib.H2 if calib.H2 is not None else np.eye(3, dtype=np.float64)
    width = img_left.shape[1]
    height = img_left.shape[0]
    rectified_left = cv.warpPerspective(
        img_left, H1, (width, height),
        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
    )
    rectified_right = cv.warpPerspective(
        img_right, H2, (width, height),
        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
    )
    return rectified_left, rectified_right