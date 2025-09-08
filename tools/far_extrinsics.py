import os, sys, math, argparse, json, time
from pathlib import Path
import numpy as np
import cv2 as cv
import yaml

# ---------- utils ----------

def load_mono_npz(path: str):
    d = np.load(path)
    # допускаем ключи как K/dist или K1/D1
    if "K" in d and "dist" in d:
        K = d["K"]; D = d["dist"]
    elif "K1" in d and "D1" in d:
        K = d["K1"]; D = d["D1"]
    else:
        raise ValueError(f"{path}: expected (K,dist) or (K1,D1) keys")
    return K.astype(np.float64), D.astype(np.float64)

def grab_one_frame(src: str, w: int, h: int):
    # Ведём себя как ffplay: TCP + длинные таймауты
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|allowed_media_types;video|stimeout;20000000|rw_timeout;20000000|max_delay;1000000|buffer_size;1048576"
    )
    cap = cv.VideoCapture(src, cv.CAP_FFMPEG if str(src).startswith("rtsp://") else cv.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    # размер выставлять для локальных камер/файлов; RTSP обычно игнорирует
    if not str(src).startswith("rtsp://"):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    # прогреем несколько кадров
    t0 = time.time()
    frame = None
    for _ in range(60):
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

def undistort_if_needed(img, K, D):
    if D is None or (np.allclose(D, 0) or D.size == 0):
        return img
    return cv.undistort(img, K, D)

def to_gray(img):
    return img if img.ndim == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def pick_detector():
    try:
        # SIFT — лучше для дальних текстур (если есть contrib)
        sift = cv.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=10)
        return sift, "SIFT"
    except Exception:
        orb = cv.ORB_create(nfeatures=8000, scaleFactor=1.2, edgeThreshold=31, patchSize=31, fastThreshold=7)
        return orb, "ORB"

def match_feats(descL, descR, use_sift: bool):
    norm = cv.NORM_L2 if use_sift else cv.NORM_HAMMING
    matcher = cv.BFMatcher(normType=norm, crossCheck=False)
    knn = matcher.knnMatch(descL, descR, k=2)
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def ransac_F(ptsL, ptsR, th_px=1.0, conf=0.999):
    F, mask = cv.findFundamentalMat(ptsL, ptsR, cv.FM_RANSAC, th_px, conf)
    if F is None or mask is None:
        raise RuntimeError("findFundamentalMat failed")
    inl = mask.ravel().astype(bool)
    return F, inl

def recover_RT_from_F(F, K1, K2, ptsL, ptsR):
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    # enforce rank-2
    S = np.array([ (S[0]+S[1])*0.5, (S[0]+S[1])*0.5, 0.0 ])
    E = U @ np.diag(S) @ Vt
    # Нормализованные координаты
    pts1n = cv.undistortPoints(ptsL.reshape(-1,1,2), K1, None)
    pts2n = cv.undistortPoints(ptsR.reshape(-1,1,2), K2, None)
    # recoverPose сам выберет «правильную» четверку (R,t)
    _, R, t, mask = cv.recoverPose(E, pts1n, pts2n)
    inl = mask.ravel().astype(bool)
    return R, t.reshape(3), inl

def stereo_rectify_full(K1,D1,K2,D2,size, R, t):
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K1, D1, K2, D2, size, R, t, flags=cv.CALIB_ZERO_DISPARITY, alpha=1.0)
    mapLx, mapLy = cv.initUndistortRectifyMap(K1, D1, R1, P1, size, cv.CV_32FC1)
    mapRx, mapRy = cv.initUndistortRectifyMap(K2, D2, R2, P2, size, cv.CV_32FC1)
    return (R1,R2,P1,P2,Q), (mapLx,mapLy,mapRx,mapRy)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Estimate stereo extrinsics by far-scene features")
    ap.add_argument("--config", required=True, help="YAML с полями left/right/width/height")
    ap.add_argument("--mono_npz_left", required=True, help="npz с (K,dist) или (K1,D1) для левой")
    ap.add_argument("--mono_npz_right", help="npz для правой (если не задан, берём левую)")
    ap.add_argument("--baseline_m", type=float, required=True, help="базис между камерами, метры (измерь рулеткой)")
    ap.add_argument("--frames", type=int, default=3, help="сколько пар кадров усреднить (1..10)")
    ap.add_argument("--out", default="intrinsics.npz", help="куда сохранить результат")
    ap.add_argument("--save_debug", default="", help="папка для дебаг-кадров/совпадений")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    left_url = str(cfg["left"]); right_url = str(cfg["right"])
    W = int(cfg.get("width", 2880)); H = int(cfg.get("height", 1620))
    size = (W, H)

    K1, D1 = load_mono_npz(args.mono_npz_left)
    if args.mono_npz_right:
        K2, D2 = load_mono_npz(args.mono_npz_right)
    else:
        K2, D2 = K1.copy(), D1.copy()

    det, name = pick_detector()
    use_sift = (name == "SIFT")

    all_ptsL = []
    all_ptsR = []

    dbg_dir = Path(args.save_debug) if args.save_debug else None
    if dbg_dir:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    # Сбор пар соответствий с нескольких кадров
    for i in range(max(1, args.frames)):
        L = grab_one_frame(left_url, W, H)
        R = grab_one_frame(right_url, W, H)

        gL = to_gray(undistort_if_needed(L, K1, D1))
        gR = to_gray(undistort_if_needed(R, K2, D2))

        # Чтобы взять «дальний фон», можно обрезать низ (порог горизонта)
        # horizon = int(0.55 * H)
        # gL = gL[:horizon, :]
        # gR = gR[:horizon, :]

        kpsL, desL = det.detectAndCompute(gL, None)
        kpsR, desR = det.detectAndCompute(gR, None)
        if desL is None or desR is None or len(kpsL) < 100 or len(kpsR) < 100:
            print(f"[{i}] too few features: L={len(kpsL) if kpsL else 0}, R={len(kpsR) if kpsR else 0}")
            continue

        good = match_feats(desL, desR, use_sift)
        if len(good) < 60:
            print(f"[{i}] too few good matches: {len(good)}")
            continue

        ptsL = np.float32([kpsL[m.queryIdx].pt for m in good])
        ptsR = np.float32([kpsR[m.trainIdx].pt for m in good])

        # опционально: грубый отбор по вертикали (эпиполярные линии ~ горизонтальны)
        # dy = np.abs(ptsL[:,1] - ptsR[:,1])
        # ok = dy < 20.0
        # ptsL, ptsR = ptsL[ok], ptsR[ok]

        F, inl = ransac_F(ptsL, ptsR, th_px=1.0, conf=0.999)
        ptsL_inl = ptsL[inl]; ptsR_inl = ptsR[inl]
        print(f"[{i}] matches={len(good)} inliers={inl.sum()}")

        if dbg_dir:
            vis = cv.drawMatches(
                gL, kpsL, gR, kpsR,
                [cv.DMatch(_imgIdx=0, _queryIdx=int(np.where(inl)[0][k]), _trainIdx=int(np.where(inl)[0][k]), _distance=0)
                 for k in range(min(150, inl.sum()))],
                None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv.imwrite(str(dbg_dir / f"matches_{i:02d}.jpg"), vis)

        all_ptsL.append(ptsL_inl)
        all_ptsR.append(ptsR_inl)

    if len(all_ptsL) == 0:
        print("No inliers collected; try more frames or adjust scene/lighting")
        sys.exit(2)

    ptsL = np.vstack(all_ptsL)
    ptsR = np.vstack(all_ptsR)

    # финальная оценка F по всем накопленным инлайнерам
    F, inl = ransac_F(ptsL, ptsR, th_px=1.0, conf=0.999)
    ptsL = ptsL[inl]; ptsR = ptsR[inl]
    print(f"TOTAL inliers: {len(ptsL)}")

    # Восстанавливаем R,t (до масштаба), затем масштабируем базисом
    R, t_unit, inl2 = recover_RT_from_F(F, K1, K2, ptsL, ptsR)
    t_unit = t_unit / (np.linalg.norm(t_unit) + 1e-9)
    t = t_unit * float(args.baseline_m)
    print("R=\n", R)
    print("t (meters, baseline applied) =", t)

    # Ректификация и карты
    (R1, R2, P1, P2, Q), (mapLx, mapLy, mapRx, mapRy) = stereo_rectify_full(K1, D1, K2, D2, size, R, t)

    # Сохраняем полноценный intrinsics.npz
    out = args.out
    np.savez_compressed(
        out,
        image_size=np.array(size, np.int32),
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=t.reshape(3,1),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy,
    )
    print(f"Saved {out}  (size={size})")

if __name__ == "__main__":
    main()
