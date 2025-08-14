# stereo_rtsp_landscape_calib.py
import os, time, math, argparse
from pathlib import Path
import cv2
import numpy as np

# ---------------------- Утиліти ----------------------

def set_ffmpeg_tcp():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

def open_rtsp(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def capture_rtsp_pairs(urlL, urlR, num_pairs=20, interval_sec=1.5,
                       timeout_sec=180.0, resize_to=None, warmup=30, save_dir=None):
    set_ffmpeg_tcp()
    capL, capR = open_rtsp(urlL), open_rtsp(urlR)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Не вдалось відкрити один із RTSP-потоків")

    for _ in range(max(0, warmup)):  # прогрів декодера
        capL.grab(); capR.grab()
        capL.retrieve(); capR.retrieve()

    pairs, t0 = [], time.time()
    next_t = time.time()

    while len(pairs) < num_pairs and (time.time() - t0) < timeout_sec:
        now = time.time()
        if now < next_t:
            time.sleep(min(0.01, next_t - now)); continue
        next_t = now + interval_sec

        capL.grab(); capR.grab()
        okL, fL = capL.retrieve()
        okR, fR = capR.retrieve()
        if not (okL and okR):
            continue

        if resize_to:
            w, h = resize_to
            fL = cv2.resize(fL, (w, h), interpolation=cv2.INTER_AREA)
            fR = cv2.resize(fR, (w, h), interpolation=cv2.INTER_AREA)

        if fL.shape[:2] != fR.shape[:2]:
            h = min(fL.shape[0], fR.shape[0])
            w = min(fL.shape[1], fR.shape[1])
            fL = cv2.resize(fL, (w, h), interpolation=cv2.INTER_AREA)
            fR = cv2.resize(fR, (w, h), interpolation=cv2.INTER_AREA)

        pairs.append((fL, fR))
        print(f"[{len(pairs)}/{num_pairs}] знято пару")

        if save_dir:
            cv2.imwrite(str(save_dir / f"raw_{len(pairs):02d}_L.jpg"), fL)
            cv2.imwrite(str(save_dir / f"raw_{len(pairs):02d}_R.jpg"), fR)

    capL.release(); capR.release()

    if len(pairs) < num_pairs:
        print(f"Увага: знято лише {len(pairs)} пар із {num_pairs}")
    return pairs

def load_intrinsics(pathK, pathD):
    if pathK and pathD and Path(pathK).exists() and Path(pathD).exists():
        K, D = np.load(pathK), np.load(pathD)
        return np.asarray(K, np.float64), np.asarray(D, np.float64)
    return None, None

# NEW: завантаження з одного NPZ для обох камер
def load_intrinsics_npz(npz_path):
    if not npz_path: return None, None
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"Не знайдено {npz_path}")
    data = np.load(p, allow_pickle=False)
    K = None; D = None
    for k in ("K", "K1", "camera_matrix", "mtx"):
        if k in data: K = data[k]
    for k in ("D", "D1", "dist", "distCoeffs", "distortion"):
        if k in data: D = data[k]
    if K is None:
        raise ValueError(f"{npz_path} не містить матрицю камери (очікував ключі K/K1/camera_matrix/mtx)")
    if D is None:
        print("[!] У файлі немає D — використовую нульову дисторсію")
        D = np.zeros((5,), dtype=np.float64)
    return np.asarray(K, np.float64), np.asarray(D, np.float64)

def make_feature_extractor(name):
    name = name.upper()
    if name == "SIFT":
        return cv2.SIFT_create(), "SIFT"
    if name == "AKAZE":
        return cv2.AKAZE_create(), "AKAZE"
    if name == "ORB":
        return cv2.ORB_create(nfeatures=4000), "ORB"
    raise ValueError("Unknown feature: " + name)

def match_keypoints(desc1, desc2, feat_name, ratio=0.75):
    if feat_name == "SIFT":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        if desc1.dtype != np.float32: desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32: desc2 = desc2.astype(np.float32)
        knn = flann.knnMatch(desc1, desc2, k=2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def enforce_rank2(F):
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    return U @ np.diag(S) @ Vt

def R_to_q(R):
    qw = math.sqrt(max(0.0, 1 + np.trace(R))) / 2.0
    qx = (R[2,1] - R[1,2])/(4*qw + 1e-12)
    qy = (R[0,2] - R[2,0])/(4*qw + 1e-12)
    qz = (R[1,0] - R[0,1])/(4*qw + 1e-12)
    return np.array([qw, qx, qy, qz], dtype=np.float64)

def q_to_R(q):
    q = q/(np.linalg.norm(q) + 1e-12)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def average_rotations(Rs):
    if not Rs: return np.eye(3)
    ref = R_to_q(Rs[0]); acc = np.zeros(4)
    for R in Rs:
        q = R_to_q(R)
        if np.dot(q, ref) < 0: q = -q
        acc += q
    return q_to_R(acc)

def unit(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v/n

def process_pair(imgL, imgR, feat_name="SIFT", ratio=0.75, ransac_thresh=1.0,
                 K1=None, D1=None, K2=None, D2=None):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    feat, fname = make_feature_extractor(feat_name)
    k1, d1 = feat.detectAndCompute(grayL, None)
    k2, d2 = feat.detectAndCompute(grayR, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        raise RuntimeError("Замало фіч")

    good = match_keypoints(d1, d2, fname, ratio)
    if len(good) < 8:
        raise RuntimeError("Замало збігів після ratio-тесту")

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh, 0.999)
    if F is None or F.shape != (3,3):
        raise RuntimeError("Не вдалося знайти F")
    inl = mask.ravel().astype(bool)
    pts1i, pts2i = pts1[inl], pts2[inl]

    R = None; t = None
    if K1 is not None and K2 is not None:
        pts1n = cv2.undistortPoints(pts1i, K1, D1) if D1 is not None else cv2.undistortPoints(pts1i, K1, None)
        pts2n = cv2.undistortPoints(pts2i, K2, D2) if D2 is not None else cv2.undistortPoints(pts2i, K2, None)
        E, _ = cv2.findEssentialMat(pts1n, pts2n, focal=1.0, pp=(0.,0.), method=cv2.RANSAC, prob=0.999, threshold=1e-3)
        if E is not None:
            _, R, t, _ = cv2.recoverPose(E, pts1n, pts2n)

    return F, R, t, pts1i, pts2i

# ---------------------- Головне ----------------------

def main():
    ap = argparse.ArgumentParser("Стерео-калібрування з двох RTSP (ландшафт) з усередненням")
    ap.add_argument("--left",  required=True, help="RTSP URL лівої камери (краще subtype=1)")
    ap.add_argument("--right", required=True, help="RTSP URL правої камери (краще subtype=1)")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--feat", choices=["SIFT","AKAZE","ORB"], default="SIFT")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac", type=float, default=1.0)
    ap.add_argument("--K1"); ap.add_argument("--D1")
    ap.add_argument("--K2"); ap.add_argument("--D2")
    ap.add_argument("--intrinsics", help="NPZ з K,D (однакові для обох камер)")  # NEW
    ap.add_argument("--out", default="stereo_rtsp_out")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(exist_ok=True)
    resize_to = (args.width, args.height) if (args.width and args.height) else None

    print("[*] Зйомка пар…")
    pairs = capture_rtsp_pairs(args.left, args.right,
                               num_pairs=args.pairs, interval_sec=args.interval,
                               timeout_sec=args.timeout, resize_to=resize_to,
                               save_dir=outdir)

    print("[*] Обчислення F / (E,R,t) по кожній парі…")

    # 1) за замовчуванням: пробуємо окремі .npy
    K1, D1 = load_intrinsics(args.K1, args.D1)
    K2, D2 = load_intrinsics(args.K2, args.D2)

    # 2) якщо задано --intrinsics, він ПЕРЕЗАПИШЕ K1/D1/K2/D2 однаковими значеннями
    if args.intrinsics:
        Kc, Dc = load_intrinsics_npz(args.intrinsics)
        K1, D1, K2, D2 = Kc, Dc, Kc, Dc
        print(f"[+] Завантажено інтрінсіки з {args.intrinsics} (спільні для обох камер)")

    Fs, Rs, Ts, inliers_accum = [], [], [], []
    for i, (fL, fR) in enumerate(pairs, 1):
        try:
            F, R, t, p1i, p2i = process_pair(
                fL, fR,
                feat_name=args.feat, ratio=args.ratio, ransac_thresh=args.ransac,
                K1=K1, D1=D1, K2=K2, D2=D2
            )
            Fs.append(F)
            if R is not None and t is not None:
                Rs.append(R); Ts.append(unit(t.reshape(-1)))
            inliers_accum.append((p1i, p2i))
            print(f"[+] Пара {i}: inliers={len(p1i)}")
        except Exception as e:
            print(f"[!] Пара {i} пропущена: {e}")

    if not Fs:
        raise RuntimeError("Не вдалося отримати жодної F — замало фіч / перевір сцени")

    F_avg = enforce_rank2(np.mean(np.stack(Fs, axis=0), axis=0))
    np.save(outdir / "F_avg.npy", F_avg)
    print("[+] Збережено F_avg.npy")

    have_metric = (K1 is not None and K2 is not None and len(Rs) > 0 and len(Ts) > 0)
    if have_metric:
        R_avg = average_rotations(Rs)
        T_dir_avg = unit(np.mean(np.stack(Ts, axis=0), axis=0))
        np.save(outdir / "R_avg.npy", R_avg)
        np.save(outdir / "t_dir_avg.npy", T_dir_avg)
        print("[+] Збережено R_avg.npy, t_dir_avg.npy")

    h, w = pairs[0][0].shape[:2]

    if have_metric:
        print("[*] stereoRectify (метрична)…")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, (w, h), R_avg, T_dir_avg.reshape(3,1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w,h), cv2.CV_32FC1)

        np.save(outdir/"R1.npy", R1); np.save(outdir/"R2.npy", R2)
        np.save(outdir/"P1.npy", P1); np.save(outdir/"P2.npy", P2)
        np.save(outdir/"Q.npy", Q)
        np.save(outdir/"map1x.npy", map1x); np.save(outdir/"map1y.npy", map1y)
        np.save(outdir/"map2x.npy", map2x); np.save(outdir/"map2y.npy", map2y)

        fL, fR = pairs[-1]
        rectL = cv2.remap(fL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(fR, map2x, map2y, cv2.INTER_LINEAR)
        cv2.imwrite(str(outdir/"rectified_metric_preview.jpg"), np.hstack([rectL, rectR]))
        print("[+] rectified_metric_preview.jpg готово")

    else:
        print("[*] stereoRectifyUncalibrated (проєктна)…")
        if not inliers_accum:
            raise RuntimeError("Немає інлаєрів для ректифікації")

        pts1_all = np.vstack([p1 for (p1, _) in inliers_accum]).reshape(-1, 2).astype(np.float32)
        pts2_all = np.vstack([p2 for (_, p2) in inliers_accum]).reshape(-1, 2).astype(np.float32)

        H1, H2, ok_val = cv2.stereoRectifyUncalibrated(pts1_all, pts2_all, F_avg, imgSize=(w, h))
        success = bool(ok_val.item()) if isinstance(ok_val, np.ndarray) else bool(ok_val)
        if not success or H1 is None or H2 is None:
            raise RuntimeError("stereoRectifyUncalibrated не вдалася (ok=False або H=None)")

        H1 = np.asarray(H1, np.float64).reshape(3,3)
        H2 = np.asarray(H2, np.float64).reshape(3,3)
        if abs(H1[2,2]) > 1e-12: H1 /= H1[2,2]
        if abs(H2[2,2]) > 1e-12: H2 /= H2[2,2]

        np.save(outdir/"H1.npy", H1); np.save(outdir/"H2.npy", H2)

        fL, fR = pairs[-1]
        rectL = cv2.warpPerspective(fL, H1, (w, h))
        rectR = cv2.warpPerspective(fR, H2, (w, h))
        cv2.imwrite(str(outdir/"rectified_projective_preview.jpg"), np.hstack([rectL, rectR]))
        print("[+] rectified_projective_preview.jpg готово")

    print(f"[✓] Готово. Вихідна папка: {outdir.resolve()}")

if __name__ == "__main__":
    main()
