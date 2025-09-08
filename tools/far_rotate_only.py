# tools/far_rotate_only.py
# Rotation-only: оценка относительного поворота между лев./прав. камерами по дальним ориентирам.
# ВАЖНО: ROI задаются в НОРМАЛИЗОВАННЫХ координатах RAW-КАДРА (до undistort)!
# Используем интринзики одной камеры (K, D) для обоих кадров (по договорённости).

import os, sys, math, argparse, time
from pathlib import Path
import numpy as np
import cv2 as cv
import yaml

# ---------- Intrinsics loader (mono) ----------
def load_intrinsics_mono(path: str):
    d = np.load(path, allow_pickle=True)
    if "K" in d and ("dist" in d or "D" in d):
        K  = d["K"]
        D  = d["dist"] if "dist" in d else d["D"]
    elif "cameraMatrix" in d and "distCoeffs" in d:
        K  = d["cameraMatrix"]
        D  = d["distCoeffs"]
    elif "K1" in d and (("D1" in d) or ("dist1" in d)):
        K  = d["K1"]
        D  = d["D1"] if "D1" in d else d["dist1"]
    else:
        raise KeyError("NPZ must contain (K,dist) or (cameraMatrix,distCoeffs) or (K1,D1).")
    return K.astype(np.float64), D.astype(np.float64)

# ---------- RTSP frame grab (минимальный лаг) ----------
def grab_one_frame(src: str, w: int, h: int):
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|allowed_media_types;video|stimeout;20000000|rw_timeout;20000000|max_delay;1000000|buffer_size;1048576"
    )
    cap = cv.VideoCapture(src, cv.CAP_FFMPEG if str(src).startswith("rtsp://") else cv.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    if not str(src).startswith("rtsp://"):
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    t0 = time.time()
    frame = None
    for _ in range(90):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        if time.time() - t0 > 5.0:
            break
        cv.waitKey(1)
    cap.release()
    if frame is None:
        raise RuntimeError(f"Failed to read a frame from {src}")
    return frame

# ---------- Features & rotation-only on unit rays (из RAW) ----------
def pick_detector():
    try:
        sift = cv.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=10)
        return sift, True
    except Exception:
        orb = cv.ORB_create(nfeatures=8000, scaleFactor=1.2, edgeThreshold=31, patchSize=31, fastThreshold=7)
        return orb, False

def detect_and_desc(det, gray, roi_poly_px=None):
    if roi_poly_px is not None:
        mask = np.zeros_like(gray, np.uint8)
        cv.fillPoly(mask, [roi_poly_px.astype(np.int32)], 255)
    else:
        mask = None
    kps, des = det.detectAndCompute(gray, mask)
    return kps, des

def match_feats(descL, descR, use_sift: bool):
    norm = cv.NORM_L2 if use_sift else cv.NORM_HAMMING
    matcher = cv.BFMatcher(normType=norm, crossCheck=False)
    knn = matcher.knnMatch(descL, descR, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def to_unit_rays_from_raw(pts_xy: np.ndarray, K: np.ndarray, D: np.ndarray):
    """RAW пиксели -> нормализованные coords через cv.undistortPoints -> единичные лучи."""
    if pts_xy.size == 0:
        return np.zeros((0,3), np.float64)
    pts = pts_xy.reshape(-1,1,2).astype(np.float64)
    und = cv.undistortPoints(pts, K, D, P=None)  # -> (x',y') в нормализованной камере
    und = und.reshape(-1,2)
    rays = np.column_stack([und, np.ones(len(und), dtype=np.float64)])
    rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12)
    return rays

def kabsch_R(u1, u2):
    H = u1.T @ u2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    return R

def ransac_rotation(u1, u2, iters=4000, thr_deg=0.35):
    N = len(u1)
    if N < 3:
        return np.eye(3), np.arange(0)
    rng = np.random.default_rng(42)
    thr = math.radians(thr_deg)
    best_R = np.eye(3); best_inl = np.array([], dtype=int)
    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        R = kabsch_R(u1[idx], u2[idx])
        ang = np.arccos(np.clip(np.sum(u2*(u1@R.T), axis=1), -1, 1))
        inl = np.where(ang < thr)[0]
        if inl.size > best_inl.size:
            best_inl, best_R = inl, R
    if best_inl.size >= 3:
        best_R = kabsch_R(u1[best_inl], u2[best_inl])
    return best_R, best_inl

def euler_zyx(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    yaw   = math.atan2(R[2,1], R[2,2])
    pitch = math.atan2(-R[2,0], sy)
    roll  = math.atan2(R[1,0], R[0,0])
    return np.degrees([yaw, pitch, roll])

def parse_norm_poly(poly_str, W, H):
    if not poly_str:
        return None
    pts=[]
    for tok in poly_str.strip().split():
        if ',' not in tok:
            continue
        x,y = tok.split(',')
        pts.append([float(x)*W, float(y)*H])
    if len(pts) < 3:
        return None
    return np.array(pts, np.float32)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Rotation-only extrinsics from far scene (RAW ROI, undistortPoints). Saves only R and summary."
    )
    ap.add_argument("--config", required=True, help="YAML with left/right/width/height")
    ap.add_argument("--intrinsics", required=True, help="intrinsics.npz: K+dist (моно файл, используем для обеих камер)")
    ap.add_argument("--frames", type=int, default=4, help="сколько пар кадров собрать и агрегировать")
    ap.add_argument("--roiL", default="", help="LEFT ROI в нормализованных RAW координатах: 'x,y x,y ...'")
    ap.add_argument("--roiR", default="", help="RIGHT ROI в нормализованных RAW координатах")
    ap.add_argument("--thr_deg", type=float, default=0.35, help="RANSAC порог угла (deg) на лучах")
    ap.add_argument("--out", default="calib/rotation_only.npz", help="куда сохранить результат (только R)")
    ap.add_argument("--save_debug", default="", help="папка для картинок c матчами (опц.)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    left = str(cfg["left"]); right = str(cfg["right"])
    W = int(cfg.get("width", 2880)); H = int(cfg.get("height", 1620))

    K, D = load_intrinsics_mono(args.intrinsics)
    print(f"[info] Loaded intrinsics from {args.intrinsics}")

    det, use_sift = pick_detector()
    roiL_px = parse_norm_poly(args.roiL, W, H)
    roiR_px = parse_norm_poly(args.roiR, W, H)

    all_u1 = []; all_u2 = []
    dbg_dir = Path(args.save_debug) if args.save_debug else None
    if dbg_dir: dbg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(max(1, args.frames)):
        frameL = grab_one_frame(left, W, H)
        frameR = grab_one_frame(right, W, H)

        gL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
        gR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

        kpsL, desL = detect_and_desc(det, gL, roiL_px)
        kpsR, desR = detect_and_desc(det, gR, roiR_px)
        if desL is None or desR is None or len(kpsL) < 8 or len(kpsR) < 8:
            print(f"[{i}] not enough keypoints (L={len(kpsL) if kpsL else 0}, R={len(kpsR) if kpsR else 0})")
            continue

        good = match_feats(desL, desR, use_sift)
        if len(good) < 12:
            print(f"[{i}] not enough matches: {len(good)}")
            continue

        ptsL = np.float32([kpsL[m.queryIdx].pt for m in good])
        ptsR = np.float32([kpsR[m.trainIdx].pt for m in good])

        # В ЕДИНИЧНЫЕ ЛУЧИ из RAW координат (учитывая дисторсию K,D одной камеры)
        u1 = to_unit_rays_from_raw(ptsL, K, D)
        u2 = to_unit_rays_from_raw(ptsR, K, D)

        all_u1.append(u1); all_u2.append(u2)

        if dbg_dir:
            vis = cv.drawMatches(
                gL, kpsL, gR, kpsR,
                sorted(good, key=lambda m: m.distance)[:200],
                None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            if roiL_px is not None:
                cv.polylines(vis, [roiL_px.astype(np.int32)], True, (0,255,0), 2, cv.LINE_AA, shift=0)
            if roiR_px is not None:
                roiR_shift = roiR_px.copy(); roiR_shift[:,0] += W
                cv.polylines(vis, [roiR_shift.astype(np.int32)], True, (0,255,0), 2, cv.LINE_AA, shift=0)
            cv.imwrite(str(dbg_dir / f"matches_{i:02d}.jpg"), vis)

    if not all_u1:
        print("No usable frames/matches; try increasing --frames or adjust ROI/lighting")
        sys.exit(2)

    U1 = np.vstack(all_u1)
    U2 = np.vstack(all_u2)

    R_est, inl = ransac_rotation(U1, U2, iters=4000, thr_deg=float(args.thr_deg))
    yaw, pitch, roll = euler_zyx(R_est)
    print(f"[R] inliers {inl.size}/{U1.shape[0]}; angles [yaw,pitch,roll] = {yaw:.3f}, {pitch:.3f}, {roll:.3f} deg")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # сохраним только поворот и сводную информацию
    np.savez_compressed(
        str(out_path),
        R=R_est,
        angles_deg=np.array([yaw, pitch, roll], dtype=np.float64),
        inliers=int(inl.size),
        total_pairs=int(U1.shape[0]),
        image_size=np.array([W, H], np.int32),
        K=K, D=D,
        roiL_norm=args.roiL, roiR_norm=args.roiR,
        method="rotation_only_rays",
        matcher=("SIFT+ratio" if use_sift else "ORB+ratio"),
        ransac_thr_deg=float(args.thr_deg)
    )
    print(f"Saved rotation-only to {out_path}")

if __name__ == "__main__":
    main()
