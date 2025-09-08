# tools/roi_clicker.py
import os, time, argparse, yaml, cv2 as cv
import numpy as np

# --- imshow_resized: берём из visualize.py, при отсутствии — делаем безопасный фолбэк
_imshow_ext = None
try:
    from visualize import imshow_resized as _imshow_ext
except Exception:
    try:
        from visual_radar.visualize import imshow_resized as _imshow_ext
    except Exception:
        _imshow_ext = None

def _fallback_imshow_resized(win, img, maxw=1600, maxh=900):
    h, w = img.shape[:2]
    s = min(maxw / w, maxh / h, 1.0)
    disp = cv.resize(img, (int(w*s), int(h*s)), interpolation=cv.INTER_AREA) if s < 1.0 else img
    cv.imshow(win, disp)
    return s, s

def imshow_resized(win, img, maxw=1600, maxh=900):
    if _imshow_ext is not None:
        try:
            r = _imshow_ext(win, img)
        except TypeError:
            try: r = _imshow_ext(win, img, maxw)
            except TypeError: r = _imshow_ext(win, img, maxw, maxh)
        if r is None:
            pass
        elif isinstance(r, (tuple, list)) and len(r) == 2:
            return float(r[0]), float(r[1])
        else:
            s = float(r); return s, s
    return _fallback_imshow_resized(win, img, maxw=maxw, maxh=maxh)

# ---------- Граббер одного кадра ----------
def grab_one_frame(src, w, h):
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
    for _ in range(120):
        ok, frame = cap.read()
        if ok and frame is not None: break
        if time.time() - t0 > 5.0: break
        cv.waitKey(1)
    cap.release()
    if frame is None:
        raise RuntimeError("Failed to read a frame")
    return frame

# ---------- Загрузка intrinsics ----------
def load_KD(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "K" in d and ("dist" in d or "D" in d):
        K  = d["K"].astype(np.float64)
        D  = (d["dist"] if "dist" in d else d["D"]).astype(np.float64)
    elif "cameraMatrix" in d and "distCoeffs" in d:
        K  = d["cameraMatrix"].astype(np.float64)
        D  = d["distCoeffs"].astype(np.float64)
    elif "K1" in d and ("D1" in d or "dist1" in d):
        K  = d["K1"].astype(np.float64)
        D  = (d["D1"] if "D1" in d else d["dist1"]).astype(np.float64)
    else:
        raise KeyError("intrinsics.npz must contain K+dist (or cameraMatrix+distCoeffs / K1+D1).")
    model  = str(d.get("model", "pinhole"))
    newK0  = d.get("newK_alpha0", None)
    newK1  = d.get("newK_alpha1", None)
    newK0  = newK0.astype(np.float64) if newK0 is not None else None
    newK1  = newK1.astype(np.float64) if newK1 is not None else None
    flags  = int(d.get("flags", 0)) if "flags" in d else 0
    return K, D, model, newK0, newK1, flags

# ---------- Карты выпрямления (rectified -> raw) и само выпрямление ----------
def compute_undistort_maps(K, D, model, newK, size_wh):
    w, h = size_wh
    if D is None or D.size == 0 or np.allclose(D, 0):
        return None, None
    if model.lower().startswith("fisheye"):
        mapx, mapy = cv.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK if newK is not None else K, (w, h), cv.CV_32FC1
        )
    else:
        mapx, mapy = cv.initUndistortRectifyMap(
            K, D, np.eye(3), newK if newK is not None else K, (w, h), cv.CV_32FC1
        )
    return mapx, mapy

def remap_with_maps(img, mapx, mapy):
    if mapx is None or mapy is None: return img
    return cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

def show_scaled(win, img, maxw, maxh, viewer):
    h, w = img.shape[:2]
    if viewer == "visualize" and _imshow_ext is not None:
        # показать через visualize, попытаться извлечь масштаб и высчитать поля
        try:
            r = _imshow_ext(win, img)
        except TypeError:
            try: r = _imshow_ext(win, img, maxw)
            except TypeError: r = _imshow_ext(win, img, maxw, maxh)
        # оценим масштаб
        if isinstance(r, (tuple, list)) and len(r) == 2:
            sx, sy = float(r[0]), float(r[1])
        elif isinstance(r, (int, float)):
            sx = sy = float(r)
        else:
            sx = sy = min(maxw / w, maxh / h, 1.0)
            cv.imshow(win, cv.resize(img, (int(w*sx), int(h*sy)), interpolation=cv.INTER_AREA) if sx < 1.0 else img)
        # поля (если окно больше изображения)
        try:
            _, _, ww, wh = cv.getWindowImageRect(win)
            disp_w, disp_h = int(round(w * sx)), int(round(h * sy))
            ox = max((ww - disp_w) // 2, 0)
            oy = max((wh - disp_h) // 2, 0)
        except Exception:
            ox = oy = 0
        return sx, sy, ox, oy
    else:
        # строгий режим: сами ресайзим БЕЗ полей → offset=(0,0)
        s = min(maxw / w, maxh / h, 1.0)
        disp = cv.resize(img, (int(w*s), int(h*s)), interpolation=cv.INTER_AREA) if s < 1.0 else img
        cv.imshow(win, disp)
        return s, s, 0, 0



# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML with left/right/width/height")
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--intrinsics", default="", help="intrinsics.npz (K+dist). If set, preview is undistorted.")
    ap.add_argument("--alpha", choices=["0", "1", "same"], default="0",
                    help="0=newK_alpha0 (кроп без полей), 1=newK_alpha1 (макс.FOV), same=исходная K.")
    ap.add_argument("--export_space", choices=["raw", "rectified", "both"], default="raw",
                    help="В каких координатах печатать ROI. По умолчанию raw (до undistort).")
    ap.add_argument("--maxw", type=int, default=1600)
    ap.add_argument("--maxh", type=int, default=900)
    ap.add_argument("--viewer", choices=["simple","visualize"], default="simple",
                help="simple: наш ресайз без полей (точное соответствие клика); visualize: использовать visualize.imshow_resized")

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    src = cfg["left"] if args.side == "left" else cfg["right"]
    W = int(cfg.get("width", 2880))
    H = int(cfg.get("height", 1620))

    raw = grab_one_frame(src, W, H)

    # --- готовим карты и превью ---
    mapx = mapy = None
    rectified = raw
    if args.intrinsics:
        try:
            K, D, model, newK0, newK1, flags = load_KD(args.intrinsics)
            use_newK = None
            if args.alpha == "0" and newK0 is not None: use_newK = newK0
            elif args.alpha == "1" and newK1 is not None: use_newK = newK1
            elif args.alpha == "same": use_newK = None
            mapx, mapy = compute_undistort_maps(K, D, model=model, newK=use_newK, size_wh=(W, H))
            rectified  = remap_with_maps(raw, mapx, mapy)
            print(f"[roi_clicker] undistort applied from {args.intrinsics} "
                  f"(model: {model} | alpha: {args.alpha})")
        except Exception as e:
            print("[roi_clicker] WARN:", e, "-- using raw frame")

    # ---------- ROI interaction with scaled display ----------
    pts_rect = []  # точки в прямоугольном (показанном) пространстве (после undistort)
    pts_raw  = []  # соответствующие точки в исходном RAW-кадре
    scale_xy = [1.0, 1.0]
    offset_xy = [0, 0]

    def rect_to_raw(px, py):
        """Перевод координаты из rectified в raw по картам mapx/mapy."""
        if mapx is None or mapy is None:
            return px, py
        xi = int(np.clip(round(px), 0, W-1))
        yi = int(np.clip(round(py), 0, H-1))
        rx = float(mapx[yi, xi]); ry = float(mapy[yi, xi])
        if not np.isfinite(rx) or not np.isfinite(ry):
            return px, py
        return rx, ry

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            sx, sy = scale_xy
            ox, oy = offset_xy
            xr = np.clip(int(round((x - ox) / sx)), 0, W-1)
            yr = np.clip(int(round((y - oy) / sy)), 0, H-1)
            rx, ry = rect_to_raw(xr, yr)
            pts_rect.append((int(xr), int(yr)))
            pts_raw.append((int(np.clip(round(rx), 0, W-1)),
                            int(np.clip(round(ry), 0, H-1))))

    win = f"roi_{args.side}"
    win_flag = cv.WINDOW_AUTOSIZE if args.viewer == "simple" else cv.WINDOW_NORMAL
    cv.namedWindow(win, win_flag)
    cv.setMouseCallback(win, on_mouse)

    while True:
        vis = rectified.copy()
        # рисуем полилинию в координатах rectified (эти координаты и видит пользователь)
        for i, p in enumerate(pts_rect):
            cv.circle(vis, p, 5, (0, 255, 0), -1)
            if i > 0:
                cv.line(vis, pts_rect[i - 1], pts_rect[i], (0, 255, 0), 2)
        if len(pts_rect) >= 3:
            cv.line(vis, pts_rect[-1], pts_rect[0], (0, 200, 0), 1)

        cv.putText(vis, "LMB=add, U=undo, Enter/Space=finish, Esc=exit",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv.LINE_AA)

        sx, sy, ox, oy = show_scaled(win, vis, args.maxw, args.maxh, args.viewer)
        scale_xy[0], scale_xy[1] = float(sx), float(sy)
        offset_xy[0], offset_xy[1] = int(ox), int(oy)
        # # fallback уточнение через размер области изображения окна
        # try:
        #     _, _, ww, wh = cv.getWindowImageRect(win)
        #     if ww > 0 and wh > 0:
        #         sx, sy = ww / W, wh / H
        # except Exception:
        #     pass
        # scale_xy[0], scale_xy[1] = float(sx), float(sy)

        k = cv.waitKey(30) & 0xFF
        if k in (13, 32):  # Enter / Space
            break
        if k == 27:  # Esc
            pts_rect.clear(); pts_raw.clear()
            break
        if k in (ord('u'), ord('U')) and pts_rect:
            pts_rect.pop(); pts_raw.pop()

    cv.destroyAllWindows()

    if len(pts_raw) < 3:
        print("Полигон не задан.")
        return

    # --- вывод ROI ---
    def _norm_list(pts):
        return " ".join([f"{x / W:.4f},{y / H:.4f}" for (x, y) in pts])

    if args.export_space in ("raw", "both"):
        print(f'{args.side.upper()} ROI (RAW coords, paste into pipeline that crops BEFORE undistort):')
        print(_norm_list(pts_raw))
    if args.export_space in ("rectified", "both"):
        print(f'{args.side.upper()} ROI (RECTIFIED coords, if your pipeline crops AFTER undistort):')
        print(_norm_list(pts_rect))

if __name__ == "__main__":
    main()
