from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import cv2 as cv


@dataclass
class RotationOnlyCalib:
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray        # right->left
    size: tuple[int,int]
    mapLx: np.ndarray
    mapLy: np.ndarray
    mapRx: np.ndarray
    mapRy: np.ndarray
    proj_mode: bool = False  # пусть остаётся False (детектор будет работать как обычно)


@dataclass
class Calibration:
    size: Tuple[int, int]                  # (w, h)
    mapxL: np.ndarray
    mapyL: np.ndarray
    mapxR: np.ndarray
    mapyR: np.ndarray

    map1L: Optional[np.ndarray] = None     # CV_16SC2
    map2L: Optional[np.ndarray] = None     # CV_32FC1
    map1R: Optional[np.ndarray] = None     # CV_16SC2
    map2R: Optional[np.ndarray] = None     # CV_32FC1

    proj_mode: bool = False                # True, если нет валидной Q → проективный фоллбэк
    Q: Optional[np.ndarray] = None         # 4x4, если есть
    baseline_m: Optional[float] = None     # для информации/логики вне


def _has_all(npz: np.lib.npyio.NpzFile, keys: list[str]) -> bool:
    return all(k in npz for k in keys)


def _maps_from_KDRP(K: np.ndarray, D: np.ndarray, R: np.ndarray, P: np.ndarray, size: Tuple[int, int]):
    w, h = size
    m1, m2 = cv.initUndistortRectifyMap(
        K.astype(np.float64), D.astype(np.float64),
        R.astype(np.float64), P.astype(np.float64),
        (w, h), cv.CV_32FC1
    )
    return m1, m2


def _maps_from_H(H: np.ndarray, size: Tuple[int, int]):
    """Строим карты для cv.remap из гомографии (rect -> src)."""
    w, h = size
    Hinv = np.linalg.inv(H.astype(np.float64))
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    ones = np.ones_like(xs)
    grid = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3)  # (N,3)
    src = grid @ Hinv.T
    src /= src[:, 2:3] + 1e-12
    mapx = src[:, 0].reshape(h, w).astype(np.float32)
    mapy = src[:, 1].reshape(h, w).astype(np.float32)
    return mapx, mapy

def _load_KD_mono(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    if "K" in d and ("dist" in d or "D" in d):
        K  = d["K"].astype(np.float64)
        D  = (d["dist"] if "dist" in d else d["D"]).astype(np.float64)
    elif "cameraMatrix" in d and "distCoeffs" in d:
        K  = d["cameraMatrix"].astype(np.float64)
        D  = d["distCoeffs"].astype(np.float64)
    elif "K1" in d and (("D1" in d) or ("dist1" in d)):
        K  = d["K1"].astype(np.float64)
        D  = (d["D1"] if "D1" in d else d["dist1"]).astype(np.float64)
    else:
        raise KeyError("intrinsics npz must contain K+dist (or cameraMatrix+distCoeffs / K1+D1)")
    return K, D

def _rotz_cw(deg: float) -> np.ndarray:
    # ПОЛОЖИТЕЛЬНЫЙ deg = ПОВОРОТ ПО ЧАСОВОЙ СТРЕЛКЕ в изображении
    a = -np.deg2rad(deg)  # минус, т.к. в камере +Z вперёд => +a = CCW
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)



def load_calibration(
    calib_dir: str,
    intrinsics_path: Optional[str],
    size: Tuple[int, int],
    baseline_m: Optional[float] = None,
) -> Calibration:
    """
    Универсальный загрузчик:
      • если есть K1/D1/K2/D2 и R,T — ПЕРЕСОБИРАЕМ карты через stereoRectify с alpha=1.0
        (сохраняем максимум обзора, без «зума»);
      • иначе читаем готовые map* из NPZ;
      • иначе строим карты из K/D/R/P или H1/H2.

    Дополнительно:
      • convertMaps(CV_16SC2) — быстрый remap;
      • жёсткие проверки Q (PR-05): нет валидной Q → proj_mode=True.
    """
    w, h = int(size[0]), int(size[1])

    # --- rotation-only: intrinsics одной камеры + R из rotation_only.npz ---
    rot_path = Path(calib_dir or ".") / "rotation_only.npz"
    intr_mono_path = Path(intrinsics_path) if intrinsics_path else Path(calib_dir or ".") / "intrinsics.npz"

    if rot_path.exists() and intr_mono_path.exists():
        K, D = _load_KD_mono(str(intr_mono_path))
        dR = np.load(str(rot_path), allow_pickle=True)
        R  = dR["R"].astype(np.float64)  # R: left -> right

        alpha = float(os.getenv("VR_RECTIFY_ALPHA", "1.0"))
        # фиктивный baseline — нужен только чтобы stereoRectify вернул согласованные R1/R2,P1/P2
        T_rect = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        R1, R2, P1, P2, _Q, _roi1, _roi2 = cv.stereoRectify(
            K, D, K, D, (w, h),
            R, T_rect,
            flags=cv.CALIB_ZERO_DISPARITY,
            alpha=alpha
        )

        # общий ролл-офсет по ЧАСОВОЙ стрелке (если задан)
        roll_deg = float(os.getenv("VR_ROLL_DEG", "0"))
        if abs(roll_deg) > 1e-6:
            Rz = _rotz_cw(roll_deg)
            R1 = Rz @ R1
            R2 = Rz @ R2
            print(f"[calib] apply common roll offset: {roll_deg:.2f} deg CW")

        # --- вместо P1/P2 можем сохранить исходный K (меньше «вогнутости»), с опц. масштабом ---
        preserve_k = os.getenv("VR_RECTIFY_PRESERVE_K", "0") == "1"
        scale = float(os.getenv("VR_RECTIFY_SCALE", "1.0"))

        if preserve_k or abs(scale - 1.0) > 1e-6:
            newK = K.astype(np.float64).copy()
            newK[0,1] = 0.0       # без skew
            newK[0,0] *= scale
            newK[1,1] *= scale
            mapxL, mapyL = cv.initUndistortRectifyMap(K, D, R1, newK, (w, h), cv.CV_32FC1)
            mapxR, mapyR = cv.initUndistortRectifyMap(K, D, R2, newK, (w, h), cv.CV_32FC1)
            print(f"[calib] preserve-K in rotation-only: scale={scale:.3f}")
        else:
            # стандарт: использовать P1/P2 от stereoRectify
            mapxL, mapyL = cv.initUndistortRectifyMap(K, D, R1, P1[:3, :3], (w, h), cv.CV_32FC1)
            mapxR, mapyR = cv.initUndistortRectifyMap(K, D, R2, P2[:3, :3], (w, h), cv.CV_32FC1)

        calb = Calibration(
            size=(w, h),
            mapxL=mapxL, mapyL=mapyL,
            mapxR=mapxR, mapyR=mapyR,
            baseline_m=float(baseline_m) if baseline_m is not None else None,
        )
        calb.proj_mode = True  # глубина не нужна
        calb.map1L, calb.map2L = cv.convertMaps(calb.mapxL, calb.mapyL, cv.CV_16SC2)
        calb.map1R, calb.map2R = cv.convertMaps(calb.mapxR, calb.mapyR, cv.CV_16SC2)
        print(f"[visual_radar.calibration] rotation-only via stereoRectify: intr={intr_mono_path.name}, R={rot_path.name}, size={w}x{h}, alpha={alpha}")
        return calb

    # 1) выбрать файл
    p = intrinsics_path or os.path.join(calib_dir or ".", "intrinsics_stereo.npz")
    if not os.path.isfile(p):
        alt = os.path.join(calib_dir or ".", "intrinsics.npz")
        if os.path.isfile(alt):
            p = alt
        else:
            raise FileNotFoundError(
                f"Calibration file not found in '{calib_dir}'. "
                "Expected intrinsics_stereo.npz or intrinsics.npz"
            )

    npz = np.load(p)

    # 2) Если в файле есть «полная» стерео-калибровка (K/D + R,T) —
    #    пересоберём R1/R2,P1/P2 с alpha=1.0 (без кропа), затем карты
    if _has_all(npz, ["K1", "D1", "K2", "D2"]) and _has_all(npz, ["R", "T"]):
        K1, D1 = npz["K1"], npz["D1"]
        K2, D2 = npz["K2"], npz["D2"]
        R, T   = npz["R"],  npz["T"]

        # Режим «чистый поворот» на основе stereo intrinsics (редкий фоллбэк)
        if os.getenv("VR_ROT_ONLY", "0") == "1" and _has_all(npz, ["K1","D1","K2","D2","R"]):
            K1, D1 = npz["K1"].astype(np.float64), npz["D1"].astype(np.float64)
            K2, D2 = npz["K2"].astype(np.float64), npz["D2"].astype(np.float64)
            R = npz["R"].astype(np.float64)
            w, h = int(size[0]), int(size[1])
            alpha = float(os.getenv("VR_RECTIFY_ALPHA", "1.0"))

            newK1, _ = cv.getOptimalNewCameraMatrix(K1, D1, (w, h), alpha, (w, h), centerPrincipalPoint=False)
            newK2, _ = cv.getOptimalNewCameraMatrix(K2, D2, (w, h), alpha, (w, h), centerPrincipalPoint=False)
            newK = 0.5 * (newK1 + newK2); newK[0,1] = 0.0

            mapxL, mapyL = cv.initUndistortRectifyMap(K1, D1, np.eye(3), newK, (w, h), cv.CV_32FC1)
            R_lr = R; R_rl = R_lr.T
            mapxR, mapyR = cv.initUndistortRectifyMap(K2, D2, R_rl, newK, (w, h), cv.CV_32FC1)
            print("[visual_radar.calibration] rotation-only: applying R^T (right->left)")

            calb = Calibration(size=(w, h), mapxL=mapxL, mapyL=mapyL, mapxR=mapxR, mapyR=mapyR)
            calb.proj_mode = True
            calb.map1L, calb.map2L = cv.convertMaps(calb.mapxL, calb.mapyL, cv.CV_16SC2)
            calb.map1R, calb.map2R = cv.convertMaps(calb.mapxR, calb.mapyR, cv.CV_16SC2)
            print("[visual_radar.calibration] rotation-only remap (undistort + R, no stereoRectify)")
            return calb

        # Стандартная stereoRectify
        alpha = float(os.getenv("VR_RECTIFY_ALPHA", "1.0"))
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
            K1.astype(np.float64), D1.astype(np.float64),
            K2.astype(np.float64), D2.astype(np.float64),
            (w, h),
            R.astype(np.float64),
            T.reshape(3, 1).astype(np.float64),
            alpha=alpha,
            flags=cv.CALIB_ZERO_DISPARITY,
        )

        # Общий ролл-офсет
        roll_deg = float(os.getenv("VR_ROLL_DEG", "0"))
        if abs(roll_deg) > 1e-6:
            Rz = _rotz_cw(roll_deg)
            R1 = Rz @ R1
            R2 = Rz @ R2
            print(f"[calib] apply common roll offset: {roll_deg:.2f} deg CW")

        # по умолчанию — карты из P1/P2
        mapxL, mapyL = _maps_from_KDRP(K1, D1, R1, P1, (w, h))
        mapxR, mapyR = _maps_from_KDRP(K2, D2, R2, P2, (w, h))
        print(f"[visual_radar.calibration] maps rebuilt via stereoRectify alpha={alpha} (keep FOV, no crop)")

        # Опция: «сохранить масштаб/центр как у исходных K»
        if os.getenv("VR_RECTIFY_PRESERVE_K", "0") == "1":
            fx = float(min(K1[0,0], K2[0,0]))
            fy = float(min(K1[1,1], K2[1,1]))
            cx = float((K1[0,2] + K2[0,2]) * 0.5)
            cy = float((K1[1,2] + K2[1,2]) * 0.5)
            newK = np.array([[fx, 0,  cx],
                             [0,  fy, cy],
                             [0,   0,  1]], dtype=np.float64)

            mapxL, mapyL = cv.initUndistortRectifyMap(K1, D1, R1, newK, (w, h), cv.CV_32FC1)
            mapxR, mapyR = cv.initUndistortRectifyMap(K2, D2, R2, newK, (w, h), cv.CV_32FC1)

            Tx = float(T[0] if T.ndim == 1 else T[0,0])
            Q = np.array([[1, 0, 0, -cx],
                          [0, 1, 0, -cy],
                          [0, 0, 0,  fx],
                          [0, 0, -1.0/Tx, 0]], dtype=np.float64)
            print(f"[visual_radar.calibration] preserve-K: newK fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        calb = Calibration(
            size=(w, h),
            mapxL=mapxL, mapyL=mapyL,
            mapxR=mapxR, mapyR=mapyR,
            baseline_m=float(baseline_m) if baseline_m is not None else None,
        )

        if Q is not None and Q.shape == (4, 4) and np.all(np.isfinite(Q)):
            calb.Q = Q
            calb.proj_mode = False
        else:
            print("[visual_radar.calibration] WARN: stereoRectify returned invalid Q → projective fallback")
            calb.proj_mode = True

    else:
        # 3) Иначе — пытаемся воспользоваться готовыми картами или собрать из K/D/R/P или H1/H2
        have_maps = _has_all(npz, ["mapxL", "mapyL", "mapxR", "mapyR"]) or _has_all(npz, ["mapLx", "mapLy", "mapRx", "mapRy"])
        have_KDRP = _has_all(npz, ["K1", "D1", "R1", "P1", "K2", "D2", "R2", "P2"])
        have_H    = _has_all(npz, ["H1", "H2"])

        if have_maps:
            mapxL = (npz["mapxL"] if "mapxL" in npz else npz["mapLx"]).astype(np.float32, copy=False)
            mapyL = (npz["mapyL"] if "mapyL" in npz else npz["mapLy"]).astype(np.float32, copy=False)
            mapxR = (npz["mapxR"] if "mapxR" in npz else npz["mapRx"]).astype(np.float32, copy=False)
            mapyR = (npz["mapyR"] if "mapyR" in npz else npz["mapRy"]).astype(np.float32, copy=False)
        elif have_KDRP:
            K1, D1, R1, P1 = npz["K1"], npz["D1"], npz["R1"], npz["P1"]
            K2, D2, R2, P2 = npz["K2"], npz["D2"], npz["R2"], npz["P2"]
            mapxL, mapyL = _maps_from_KDRP(K1, D1, R1, P1, (w, h))
            mapxR, mapyR = _maps_from_KDRP(K2, D2, R2, P2, (w, h))
            print("[visual_radar.calibration] maps built from K/D/R/P (as is)")
        elif have_H:
            H1, H2 = npz["H1"], npz["H2"]
            mapxL, mapyL = _maps_from_H(H1, (w, h))
            mapxR, mapyR = _maps_from_H(H2, (w, h))
            print("[visual_radar.calibration] maps built from H1/H2 (projective)")
        else:
            raise KeyError(
                "NPZ does not contain rectification maps nor calibration matrices.\n"
                "Expected one of:\n"
                "  - mapxL/mapyL/mapxR/mapyR (or mapLx/mapLy/mapRx/mapRy), or\n"
                "  - K1/D1/R1/P1 and K2/D2/R2/P2, or\n"
                "  - H1/H2, or\n"
                "  - K1/D1/K2/D2 + R,T (preferred)."
            )

        for name, m in (("mapxL", mapxL), ("mapyL", mapyL), ("mapxR", mapxR), ("mapyR", mapyR)):
            if m.shape != (h, w):
                raise ValueError(f"{name} has shape {m.shape}, expected {(h, w)}")

        calb = Calibration(
            size=(w, h),
            mapxL=mapxL, mapyL=mapyL,
            mapxR=mapxR, mapyR=mapyR,
            baseline_m=float(baseline_m) if baseline_m is not None else None,
        )

        if "Q" in npz and npz["Q"].shape == (4, 4) and np.all(np.isfinite(npz["Q"])):
            calb.Q = npz["Q"]
            calb.proj_mode = False
        else:
            print("[visual_radar.calibration] WARN: no valid Q → using projective fallback (no 3D depth).")
            calb.proj_mode = True

    # 4) быстрый формат карт для remap
    calb.map1L, calb.map2L = cv.convertMaps(calb.mapxL, calb.mapyL, cv.CV_16SC2)
    calb.map1R, calb.map2R = cv.convertMaps(calb.mapxR, calb.mapyR, cv.CV_16SC2)

    print(f"[visual_radar.calibration] loaded {os.path.basename(p)} for size={w}x{h} proj_mode={calb.proj_mode}")
    return calb


def rectified_pair(calb: Calibration, frameL: np.ndarray, frameR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Быстрый remap с мягким приведением размеров к ожидаемым."""
    w, h = calb.size
    if frameL.shape[:2] != (h, w):
        frameL = cv.resize(frameL, (w, h), interpolation=cv.INTER_LINEAR)
    if frameR.shape[:2] != (h, w):
        frameR = cv.resize(frameR, (w, h), interpolation=cv.INTER_LINEAR)

    Lr = cv.remap(frameL, calb.map1L, calb.map2L, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    Rr = cv.remap(frameR, calb.map1R, calb.map2R, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return Lr, Rr
