from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2 as cv
import logging

log = logging.getLogger("visual_radar.calibration")

try:
    from visual_radar.io import FFmpegRTSP_MJPEG  # noqa: F401
except Exception:
    FFmpegRTSP_MJPEG = None  


@dataclass
class Calibration:
    mode: str  
    size_ref: Tuple[int, int]  

    proj_mode: bool = False

    H1: Optional[np.ndarray] = None
    H2: Optional[np.ndarray] = None

    map1x: Optional[np.ndarray] = None  # CV_32FC1
    map1y: Optional[np.ndarray] = None  # CV_32FC1
    map2x: Optional[np.ndarray] = None  # CV_32FC1
    map2y: Optional[np.ndarray] = None  # CV_32FC1

    map1L_s16: Optional[np.ndarray] = None  # CV_16SC2
    map2L_u16: Optional[np.ndarray] = None  # CV_16UC1
    map1R_s16: Optional[np.ndarray] = None  # CV_16SC2
    map2R_u16: Optional[np.ndarray] = None  # CV_16UC1

    Q: Optional[np.ndarray] = None

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
    width, height = frame_size

    newK1, _ = cv.getOptimalNewCameraMatrix(K1, D1, (width, height), alpha)
    newK2, _ = cv.getOptimalNewCameraMatrix(K2, D2, (width, height), alpha)

    R1, R2, P1, P2, Q, _roi1, _roi2 = cv.stereoRectify(
        newK1, D1, newK2, D2, (width, height), R, T,
        flags=cv.CALIB_ZERO_DISPARITY, alpha=alpha
    )

    # 1) Fast maps
    map1L_s16, map2L_u16 = cv.initUndistortRectifyMap(newK1, D1, R1, P1, (width, height), cv.CV_16SC2)
    map1R_s16, map2R_u16 = cv.initUndistortRectifyMap(newK2, D2, R2, P2, (width, height), cv.CV_16SC2)

    # 2) Float maps
    map1x, map1y = cv.initUndistortRectifyMap(newK1, D1, R1, P1, (width, height), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(newK2, D2, R2, P2, (width, height), cv.CV_32FC1)

    return Calibration(
        mode="metric_maps",
        size_ref=(width, height),
        proj_mode=False,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        map1L_s16=map1L_s16, map2L_u16=map2L_u16,
        map1R_s16=map1R_s16, map2R_u16=map2R_u16,
        Q=Q
    )


def _proj_identity(frame_size: Tuple[int, int]) -> Calibration:
    width, height = frame_size
    H1 = np.eye(3, dtype=np.float64)
    H2 = np.eye(3, dtype=np.float64)
    log.warning(
        "[calibration] using projective fallback (H1/H2). "
        "Stereo pairing / 3D depth MUST be disabled in this mode."
    )
    return Calibration(mode="proj", size_ref=(width, height), proj_mode=True, H1=H1, H2=H2, Q=None)


def _load_intrinsics(data: Dict[str, Any], frame_size: Tuple[int, int]) -> Optional[Calibration]:
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

    if intrinsics is not None:
        data = _try_load_npz(Path(intrinsics))
        calib = _load_intrinsics(data, frame_size)
        if calib:
            return calib

    npz_path = Path(calib_dir) / "intrinsics.npz"
    data = _try_load_npz(npz_path)
    calib = _load_intrinsics(data, frame_size)
    if calib:
        return calib

    return _proj_identity(frame_size)


def rectified_pair(
    calib: Calibration,
    img_left: np.ndarray,
    img_right: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if img_left is None or img_right is None:
        raise ValueError("rectified_pair: input images must be non-None")

    hL, wL = img_left.shape[:2]
    hR, wR = img_right.shape[:2]
    if (wL, hL) != (wR, hR):
        raise ValueError(
            f"rectified_pair: left/right size mismatch: L={(wL,hL)} R={(wR,hR)}. "
            "Frames must be the same size BEFORE rectification."
        )

    if calib.mode == "metric_maps" and (calib.map1x is not None or calib.map1L_s16 is not None):
        width, height = wL, hL
        ref_width, ref_height = calib.size_ref

        if (width, height) == (ref_width, ref_height) and calib.map1L_s16 is not None:
            rectified_left = cv.remap(
                img_left, calib.map1L_s16, calib.map2L_u16,
                interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )
            rectified_right = cv.remap(
                img_right, calib.map1R_s16, calib.map2R_u16,
                interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )
            return rectified_left, rectified_right

        if calib._cache_size != (width, height):
            map1x = cv.resize(calib.map1x, (width, height), interpolation=cv.INTER_LINEAR)
            map1y = cv.resize(calib.map1y, (width, height), interpolation=cv.INTER_LINEAR)
            map2x = cv.resize(calib.map2x, (width, height), interpolation=cv.INTER_LINEAR)
            map2y = cv.resize(calib.map2y, (width, height), interpolation=cv.INTER_LINEAR)
            calib._cache_size = (width, height)
            calib._cache_maps = (map1x, map1y, map2x, map2y)

            if calib.Q is not None and (width, height) != (ref_width, ref_height):
                log.warning(
                    "[calibration] Q matrix is defined for size %s, but current size is %s — "
                    "dropping Q to avoid biased depth. Recompute intrinsics for this size.",
                    (ref_width, ref_height), (width, height)
                )
                calib.Q = None  

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

    H1 = calib.H1 if calib.H1 is not None else np.eye(3, dtype=np.float64)
    H2 = calib.H2 if calib.H2 is not None else np.eye(3, dtype=np.float64)
    rectified_left = cv.warpPerspective(
        img_left, H1, (wL, hL),
        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
    )
    rectified_right = cv.warpPerspective(
        img_right, H2, (wR, hR),
        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
    )
    return rectified_left, rectified_right


def open_mjpeg_for_calibration(
    url: str,
    frame_size: Tuple[int, int],
    ffmpeg: str = "ffmpeg",
    quality: int = 6,
    threads: int = 3,
    read_timeout: float = 1.0,
):
    if FFmpegRTSP_MJPEG is None:
        raise RuntimeError("FFmpegRTSP_MJPEG is unavailable — check that visual_radar.io is importable.")
    width, height = map(int, frame_size)
    return FFmpegRTSP_MJPEG(
        url=url,
        width=width,
        height=height,
        use_tcp=True,
        ffmpeg_cmd=ffmpeg,
        q=int(quality),
        threads=int(threads),
        read_timeout=float(read_timeout),
    )

def collect_calibration_frames(
    url: str,
    frame_size: Tuple[int, int],
    n_frames: int = 30,
    ffmpeg: str = "ffmpeg",
    quality: int = 6,
    threads: int = 3,
    read_timeout: float = 1.0,
) -> List[np.ndarray]:
    reader = open_mjpeg_for_calibration(
        url=url,
        frame_size=frame_size,
        ffmpeg=ffmpeg,
        quality=quality,
        threads=threads,
        read_timeout=read_timeout,
    )
    frames: List[np.ndarray] = []
    try:
        while len(frames) < n_frames:
            ok, frame, _ = reader.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
    finally:
        try:
            reader.release()
        except Exception:
            pass
    return frames
