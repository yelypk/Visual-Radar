
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import cv2 as cv

@dataclass
class Calibration:
    mode: str  # "proj", "metric_maps", "metric_compute"
    H1: Optional[np.ndarray] = None
    H2: Optional[np.ndarray] = None
    map1x: Optional[np.ndarray] = None
    map1y: Optional[np.ndarray] = None
    map2x: Optional[np.ndarray] = None
    map2y: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    size_ref: Optional[Tuple[int,int]] = None  # (w,h) maps reference

def _maps_exist(folder: Path):
    return all((folder/f).exists() for f in ["map1x.npy","map1y.npy","map2x.npy","map2y.npy","Q.npy"])

def load_calibration(calib_dir: Path,
                     intrinsics_npz: Optional[Path],
                     frame_size: Tuple[int,int],
                     baseline_m: Optional[float]=None) -> Calibration:
    w,h = frame_size
    if calib_dir and _maps_exist(calib_dir):
        map1x = np.load(calib_dir/"map1x.npy")
        map1y = np.load(calib_dir/"map1y.npy")
        map2x = np.load(calib_dir/"map2x.npy")
        map2y = np.load(calib_dir/"map2y.npy")
        Q     = np.load(calib_dir/"Q.npy")
        size_ref = (map1x.shape[1], map1x.shape[0])
        print("[*] Calibration: metric (maps from disk).")
        return Calibration(mode="metric_maps", map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y, Q=Q, size_ref=size_ref)

    if intrinsics_npz and intrinsics_npz.exists():
        data = np.load(intrinsics_npz, allow_pickle=True)
        K1 = data.get("K1") or data.get("cameraMatrix1"); K2 = data.get("K2") or data.get("cameraMatrix2")
        D1 = data.get("D1") or data.get("distCoeffs1");  D2 = data.get("D2") or data.get("distCoeffs2")
        R  = data.get("R"); T = data.get("T")
        if R is None:
            R = np.eye(3, dtype=np.float64)
        if T is None and baseline_m is not None:
            T = np.array([[-baseline_m, 0, 0]], dtype=np.float64).T
        assert K1 is not None and K2 is not None and D1 is not None and D2 is not None and T is not None,             "intrinsics.npz must contain K1,K2,D1,D2 and either T or provide --baseline"

        size = (w,h)
        R1,R2,P1,P2,Q,roi1,roi2 = cv.stereoRectify(K1,K2,D1,D2,size,R,T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)
        map1x,map1y = cv.initUndistortRectifyMap(K1,D1,R1,P1,size,cv.CV_32FC1)
        map2x,map2y = cv.initUndistortRectifyMap(K2,D2,R2,P2,size,cv.CV_32FC1)
        print("[*] Calibration: metric (computed from intrinsics).")
        return Calibration(mode="metric_compute", map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y, Q=Q, size_ref=size)

    print("[!] Calibration: projective fallback (no metric depth).")
    H1 = np.eye(3, dtype=np.float64); H2 = np.eye(3, dtype=np.float64)
    return Calibration(mode="proj", H1=H1, H2=H2, Q=None, size_ref=frame_size)

def rectified_pair(calib: Calibration, imgL, imgR):
    if calib.mode.startswith("metric"):
        h,w = imgL.shape[:2]
        mw,mh = calib.size_ref
        if (w,h) != (mw,mh):
            map1x = cv.resize(calib.map1x, (w,h), interpolation=cv.INTER_LINEAR)
            map1y = cv.resize(calib.map1y, (w,h), interpolation=cv.INTER_LINEAR)
            map2x = cv.resize(calib.map2x, (w,h), interpolation=cv.INTER_LINEAR)
            map2y = cv.resize(calib.map2y, (w,h), interpolation=cv.INTER_LINEAR)
        else:
            map1x,map1y,map2x,map2y = calib.map1x, calib.map1y, calib.map2x, calib.map2y
        rL = cv.remap(imgL, map1x, map1y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        rR = cv.remap(imgR, map2x, map2y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        return rL,rR
    else:
        rL = cv.warpPerspective(imgL, calib.H1, (imgL.shape[1], imgL.shape[0]))
        rR = cv.warpPerspective(imgR, calib.H2, (imgR.shape[1], imgR.shape[0]))
        return rL,rR
