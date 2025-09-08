# tools/mono_calibrate_adv.py
import os, sys, glob, argparse, time, pathlib, shutil
import numpy as np
import cv2 as cv

# ========= Project IO: быстрый RTSP ридер =========
try:
    from visual_radar.io import open_stream
except Exception:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from visual_radar.io import open_stream

# ========= imshow_resized (из visualize.py) + фолбэк =========
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
            try:
                r = _imshow_ext(win, img, maxw)
            except TypeError:
                r = _imshow_ext(win, img, maxw, maxh)
        if r is None:
            return 1.0, 1.0
        if isinstance(r, (tuple, list)) and len(r) == 2:
            return float(r[0]), float(r[1])
        return float(r), float(r)
    return _fallback_imshow_resized(win, img, maxw=maxw, maxh=maxh)

# ========= Детектор шахматки, предобработка и метрики =========
def preprocess_gray(gray, pre_eq=False, pre_blur=0):
    g = gray
    if pre_eq:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
    if pre_blur and (pre_blur % 2 == 1):
        g = cv.GaussianBlur(g, (pre_blur, pre_blur), 0)
    return g

def detect_chess(gray, board, use_sb=False):
    if use_sb and hasattr(cv, "findChessboardCornersSB"):
        flags = (cv.CALIB_CB_NORMALIZE_IMAGE |
                 cv.CALIB_CB_EXHAUSTIVE |
                 cv.CALIB_CB_ACCURACY |
                 cv.CALIB_CB_LARGER)
        ok, corners = cv.findChessboardCornersSB(gray, board, flags=flags)
        return ok, corners
    flags = (cv.CALIB_CB_ADAPTIVE_THRESH |
             cv.CALIB_CB_NORMALIZE_IMAGE |
             cv.CALIB_CB_FILTER_QUADS)
    return cv.findChessboardCorners(gray, board, flags=flags)

def refine_corners(gray, corners):
    return cv.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

def _bbox_metrics(corners, W, H):
    xy = corners.reshape(-1,2)
    x0, y0 = xy[:,0].min(), xy[:,1].min()
    x1, y1 = xy[:,0].max(), xy[:,1].max()
    margin = min(x0, y0, W-1-x1, H-1-y1)
    w, h = (x1-x0), (y1-y0)
    cov = (w*h) / float(W*H)
    short = min(w, h)
    return (x0,y0,x1,y1), margin, cov, short

# ========= Проверка набора изображений без калибровки =========
def check_images_by_glob(images_glob, boards, args):
    paths = sorted(glob.glob(images_glob))
    if not paths:
        print(f"[check] нет файлов по маске: {images_glob}")
        return
    if args.dump_ok:   os.makedirs(args.dump_ok, exist_ok=True)
    if args.dump_fail: os.makedirs(args.dump_fail, exist_ok=True)

    ok_cnt = 0
    per_board_ok = {b: 0 for b in boards}
    total = 0
    bad_examples = []

    for p in paths:
        im = cv.imread(p, cv.IMREAD_COLOR)
        if im is None:
            continue
        total += 1
        H, W = im.shape[:2]
        g0 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        g  = preprocess_gray(g0, pre_eq=args.pre_eq, pre_blur=args.pre_blur)

        matched = None
        corners_best = None
        # пробуем все шаблоны
        for b in boards:
            ok_b, c_b = detect_chess(g, b, use_sb=args.use_sb)
            if ok_b and c_b is not None and int(c_b.shape[0]) == b[0]*b[1]:
                matched, corners_best = b, c_b
                break
        note, color = "FAIL", (0,0,255)
        ok_pass = False
        if matched is not None:
            (x0,y0,x1,y1), margin, cov, short = _bbox_metrics(corners_best, W, H)
            if margin < args.min_margin_px:
                note = f"FAIL MARGIN {int(margin)}px"
            elif (short < args.min_short_px) and (cov < args.min_coverage):
                note = f"FAIL SIZE s={int(short)}px area={cov:.2f}"
            else:
                ok_pass = True
                ok_cnt += 1
                per_board_ok[matched] += 1
                note, color = f"OK {matched[0]}x{matched[1]}", (0,255,0)
        else:
            if len(bad_examples) < 12:
                bad_examples.append(p)

        if args.check_preview:
            vis = im.copy()
            if matched is not None:
                cv.drawChessboardCorners(vis, matched, corners_best, True)
                (x0,y0,x1,y1), margin, cov, short = _bbox_metrics(corners_best, W, H)
                cv.rectangle(vis, (int(x0),int(y0)), (int(x1),int(y1)), (255,0,0), 2)
                cv.putText(vis, f"margin={int(margin)}px short={int(short)} area={cov:.2f}",
                           (20,80), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2, cv.LINE_AA)
            cv.putText(vis, note, (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)
            imshow_resized("check", vis, maxw=args.maxw, maxh=args.maxh)
            if (cv.waitKey(10) & 0xFF) == 27:
                break

        if args.dump_ok and ok_pass:
            shutil.copy2(p, os.path.join(args.dump_ok, os.path.basename(p)))
        if args.dump_fail and not ok_pass:
            shutil.copy2(p, os.path.join(args.dump_fail, os.path.basename(p)))

    if args.check_preview:
        cv.destroyAllWindows()
    print(f"[check] total={total}, ok={ok_cnt}")
    print("[check] per-board:", ", ".join([f"{m}x{n}={per_board_ok[(m,n)]}" for (m,n) in boards]))
    if bad_examples:
        print("[check] примеры FAIL:")
        for p in bad_examples:
            print("  -", p)

# ========= Утилиты =========
def parse_size_arg(size_str):
    if not size_str:
        return None, None
    try:
        w, h = map(int, size_str.split(","))
        return w, h
    except Exception:
        return None, None

# ========= Главная функция =========
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images_glob", help="напр. data/calib/*.jpg")
    src.add_argument("--rtsp", help="RTSP URL камеры")

    ap.add_argument("--board", default="9x6", help="внутренние углы MxN (НЕ клетки!)")
    ap.add_argument("--board_squares", default="", help="если удобнее: клетки WxH, преобразуется в (W-1)x(H-1) углов")
    ap.add_argument("--boards", default="",
                    help="Список внутренних углов через запятую, напр. '8x6,9x6,9x7,7x8'. "
                         "Если задан, заменяет --board/--board_squares и пробует каждый по очереди.")
    ap.add_argument("--square_mm", type=float, default=30.0, help="размер клетки в мм")
    ap.add_argument("--size", default="", help="W,H при необходимости зафиксировать размер (иначе по файлам)")
    ap.add_argument("--frames", type=int, default=40, help="сколько УСПЕШНЫХ кадров собрать (онлайн)")
    ap.add_argument("--out", default="intrinsics.npz", help="куда сохранить результат")
    ap.add_argument("--alpha_preview", type=float, default=0.0, help="0..1 предпросмотра (alpha=0 без полей)")
    ap.add_argument("--use_sb", action="store_true", help="использовать findChessboardCornersSB")

    # предпросмотр
    ap.add_argument("--maxw", type=int, default=1600, help="макс. ширина окна предпросмотра")
    ap.add_argument("--maxh", type=int, default=900, help="макс. высота окна предпросмотра")
    ap.add_argument("--no_preview", action="store_true", help="не показывать окно (макс. скорость)")

    # проектный ридер
    ap.add_argument("--reader", choices=["opencv","ffmpeg_mjpeg"], default="ffmpeg_mjpeg")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--mjpeg_q", type=int, default=6)
    ap.add_argument("--ff_threads", type=int, default=3)
    ap.add_argument("--cap_buffersize", type=int, default=1)
    ap.add_argument("--read_timeout", type=float, default=0.25)

    # сбор снимков
    ap.add_argument("--save_dir", default="", help="папка для снимков с камеры")
    ap.add_argument("--save_all", action="store_true", help="сохранять каждый N-й кадр в save_dir")
    ap.add_argument("--save_stride", type=int, default=3, help="шаг для --save_all")
    ap.add_argument("--snap_only", action="store_true",
                    help="только окно и ручные снимки (C/S), без детекции. Требует --save_dir")

    # режим проверки
    ap.add_argument("--check_only", action="store_true",
                    help="только проверить распознавание (без калибровки)")
    ap.add_argument("--check_preview", action="store_true",
                    help="показывать окно при проверке")
    ap.add_argument("--try_flip", action="store_true",
                    help="дополнительно проверять NxM ↔ MxN (игнорируется, если задан --boards)")

    # критерии качества для проверки
    ap.add_argument("--pre_eq", action="store_true", help="CLAHE перед детекцией")
    ap.add_argument("--pre_blur", type=int, default=0, help="GaussianBlur ядро (нечётное). 0=выкл")
    ap.add_argument("--min_margin_px", type=int, default=8,
                    help="мин. зазор от крайнего ВНУТРЕННЕГО угла до края кадра")
    ap.add_argument("--min_coverage", type=float, default=0.18,
                    help="мин. доля площади bbox шахматки от кадра")
    ap.add_argument("--min_short_px", type=int, default=200,
                    help="мин. короткая сторона bbox шахматки в пикселях")

    # раскладка по папкам при проверке
    ap.add_argument("--dump_ok", default="", help="папка для OK-кадров (--check_only)")
    ap.add_argument("--dump_fail", default="", help="папка для FAIL-кадров (--check_only)")

    # калибровочная модель/проходы
    ap.add_argument("--two_pass", action="store_true",
                    help="Сначала стабилизирующий проход (фиксируем аспект), затем уточнение")
    ap.add_argument("--model", choices=["standard","rational","full"], default="rational",
                    help="Набор коэффициентов дисторсии: standard(k1-3,p1-2), rational(k1-6,p1-2), full(+thin_prism+tilted)")
    ap.add_argument("--fix_aspect", action="store_true",
                    help="В 1-м проходе зафиксировать fy/fx (аспект=1.0)")

    args = ap.parse_args()

    # преобразование из количества клеток (если указано)
    if args.board_squares:
        sw, sh = map(int, args.board_squares.lower().split("x"))
        args.board = f"{sw-1}x{sh-1}"
        print(f"[info] board set from cells {sw}x{sh} → inner corners {args.board}")

    # базовый узор из --board
    M, N = map(int, args.board.lower().split("x"))
    base_board = (M, N)

    # --- парсинг набора узоров ---
    def _parse_board(tok: str):
        m, n = map(int, tok.strip().lower().split("x"))
        return (m, n)
    if args.boards:
        boards = [_parse_board(tok) for tok in args.boards.split(",") if tok.strip()]
        print(f"[info] multiple boards: {', '.join([f'{m}x{n}' for m,n in boards])}")
    else:
        boards = [base_board]

    # подготовка objp для каждого узора
    objp_by_board = {}
    for (Mm, Nn) in boards:
        objp = np.zeros((Mm*Nn, 3), np.float32)
        objp[:, :2] = np.mgrid[0:Mm, 0:Nn].T.reshape(-1, 2) * args.square_mm
        objp_by_board[(Mm, Nn)] = objp

    target_w, target_h = parse_size_arg(args.size)

    # ---------- OFFLINE: по папке изображений ----------
    frames = []
    frame_paths = []
    if args.images_glob:
        for p in sorted(glob.glob(args.images_glob)):
            im = cv.imread(p, cv.IMREAD_COLOR)
            if im is not None:
                frames.append(im)
                frame_paths.append(p)
        # режим проверки без калибровки
        if args.check_only:
            check_images_by_glob(args.images_glob, boards, args)
            return

    # ---------- ONLINE: поток с камеры ----------
    elif args.rtsp:
        if args.snap_only and not args.save_dir:
            raise RuntimeError("--snap_only требует указать --save_dir")
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)

        w_for_reader = target_w if target_w else 0
        h_for_reader = target_h if target_h else 0
        stream = open_stream(
            args.rtsp, w_for_reader, h_for_reader,
            reader=args.reader, ffmpeg=args.ffmpeg,
            mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads,
            cap_buffersize=args.cap_buffersize, read_timeout=args.read_timeout,
        )
        print(f"[calib] reader={args.reader} (buffersize={args.cap_buffersize}, timeout={args.read_timeout}s)")
        print("[calib] S/C=save snapshot (в режиме snap_only — снимок без детекции), Q=quit")

        ok_cnt, frame_i = 0, 0
        try:
            while True:
                ok, im, _ts = stream.read()
                if not ok or im is None:
                    if not args.no_preview and (cv.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
                        break
                    continue

                # дренаж буфера для снижения лага
                for _ in range(4):
                    ok2, im2, _ = stream.read()
                    if not ok2 or im2 is None:
                        break
                    im = im2

                frame_i += 1

                # --- SNAP-ONLY: окно + ручные снимки, без детекции ---
                if args.snap_only:
                    vis = im.copy()
                    cv.putText(vis, "SNAP-ONLY | C/S=save, Q=quit",
                               (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv.LINE_AA)
                    if not args.no_preview:
                        imshow_resized("calib", vis, maxw=args.maxw, maxh=args.maxh)
                        k = cv.waitKey(1) & 0xFF
                    else:
                        k = 0
                    if k in (ord('c'), ord('C'), ord('s'), ord('S')):
                        if args.save_dir:
                            fn = os.path.join(args.save_dir, f"calib_{int(time.time()*1000)}.jpg")
                            cv.imwrite(fn, im, [int(cv.IMWRITE_JPEG_QUALITY), 95])
                            print(f"[snap] saved {fn}")
                    if k in (ord('q'), ord('Q')):
                        break
                    continue  # никакой детекции

                # --- обычный режим онлайн: детекция и сохранение УСПЕШНЫХ ---
                gray0 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                gray  = preprocess_gray(gray0, pre_eq=args.pre_eq, pre_blur=args.pre_blur)

                matched = None
                corners_best = None
                for b in boards:
                    ok_b, c_b = detect_chess(gray, b, use_sb=args.use_sb)
                    if ok_b and c_b is not None and int(c_b.shape[0]) == b[0]*b[1]:
                        matched, corners_best = b, c_b
                        break

                if not args.no_preview:
                    vis = im.copy()
                    if matched is not None:
                        cv.drawChessboardCorners(vis, matched, corners_best, True)
                        cv.putText(vis, f"{ok_cnt}/{args.frames} (board {matched[0]}x{matched[1]})",
                                   (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv.LINE_AA)
                    else:
                        cv.putText(vis, f"{ok_cnt}/{args.frames}  (s/S=save, c/C=snap, q/Q=quit)",
                                   (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv.LINE_AA)
                    imshow_resized("calib", vis, maxw=args.maxw, maxh=args.maxh)
                    k = cv.waitKey(1) & 0xFF
                else:
                    k = 0

                # ручной снимок в папку
                if k in (ord('c'), ord('C')) and args.save_dir:
                    fn = os.path.join(args.save_dir, f"calib_{int(time.time()*1000)}.jpg")
                    cv.imwrite(fn, im, [int(cv.IMWRITE_JPEG_QUALITY), 95])
                    print(f"[snap] saved {fn}")

                # сохранить кадр ДЛЯ калибровки (только если углы найдены под какой-то шаблон)
                if k in (ord('s'), ord('S')):
                    if matched is not None:
                        frames.append(im.copy())
                        if 'frame_paths' not in locals():
                            frame_paths = []
                        frame_paths.append(f"[online:{matched[0]}x{matched[1]}]")
                        ok_cnt += 1
                        print(f"[calib] saved {ok_cnt}/{args.frames} with board {matched[0]}x{matched[1]}")
                    else:
                        print("[calib] шахматка НЕ найдена под ни один шаблон — кадр не сохранён")

                if (not args.no_preview and k in (ord('q'), ord('Q'))) or (ok_cnt >= args.frames > 0):
                    break
        finally:
            try:
                stream.release()
            except Exception:
                pass
            if not args.no_preview:
                cv.destroyAllWindows()
    else:
        raise RuntimeError("Нужно указать либо --images_glob, либо --rtsp.")

    # ---------- Если мы только снимали snap_only — здесь выходим ----------
    if args.snap_only:
        print(f"[snap] Done. Snaps in: {args.save_dir}")
        print(f"[next] Калибруем по папке: --images_glob \"{args.save_dir}/*.jpg\" "
              f"--boards \"{','.join([f'{m}x{n}' for m,n in boards])}\" --square_mm {args.square_mm} --use_sb")
        return

    # ---------- Проверка наличия кадров для калибровки ----------
    if not frames:
        raise RuntimeError(
            "Нет кадров для калибровки: не сохранено ни одного УСПЕШНОГО кадра (s/S), "
            "или углы шахматки ни разу не были найдены. "
            "Соберите офлайн-снимки (--save_dir, 'S/C' в snap_only), затем запустите с --images_glob."
        )

    # ---------- Согласуем imageSize с реальным размером файлов ----------
    H0, W0 = frames[0].shape[:2]
    if target_w and target_h and (W0 != target_w or H0 != target_h):
        print(f"[warn] изображения имеют {W0}x{H0}, а --size {target_w}x{target_h}; "
              f"использую фактический {W0}x{H0}")
        target_w, target_h = W0, H0

    # ---------- Соберём точки углов по всем кадрам (мульти-шаблон) ----------
    objpoints, imgpoints = [], []
    used_stats = {b: 0 for b in boards}

    for idx, im in enumerate(frames):
        gray0 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        gray  = preprocess_gray(gray0, pre_eq=args.pre_eq, pre_blur=args.pre_blur)

        matched = None
        corners_best = None
        for b in boards:
            ok_b, c_b = detect_chess(gray, b, use_sb=args.use_sb)
            if ok_b and c_b is not None and int(c_b.shape[0]) == b[0]*b[1]:
                matched, corners_best = b, c_b
                break

        if matched is None:
            src_name = frame_paths[idx] if idx < len(frame_paths) else f"frame#{idx}"
            print(f"[skip] {src_name}: ни один шаблон не подошёл ({', '.join([f'{m}x{n}' for m,n in boards])})")
            continue

        corners_ref = refine_corners(gray, corners_best)
        objpoints.append(objp_by_board[matched].copy())
        imgpoints.append(corners_ref.reshape(-1,2))
        used_stats[matched] += 1

    print("[calib] used frames per board:",
          ", ".join([f"{m}x{n}={used_stats[(m,n)]}" for (m,n) in boards]))

    if len(objpoints) < 6:
        raise RuntimeError(f"Недостаточно успешных кадров: {len(objpoints)} (<6). "
                           "Добавьте крупные ракурсы, загоняйте шахматку в углы, уменьшите блики, используйте --use_sb.")

    # ---------- Размер кадра ----------
    if target_w and target_h:
        W, H = target_w, target_h
    else:
        H, W = frames[0].shape[:2]

    # ---------- Выбор модели дисторсии ----------
    if args.model == "standard":
        flags_base = 0
    elif args.model == "rational":
        flags_base = cv.CALIB_RATIONAL_MODEL
    else:  # full
        flags_base = (cv.CALIB_RATIONAL_MODEL | cv.CALIB_THIN_PRISM_MODEL | cv.CALIB_TILTED_MODEL)

    # стартовая матрица — помогает стабилизировать фокусы
    initK = cv.initCameraMatrix2D(objpoints, imgpoints, (W, H), 1)

    def _calib(flags, K0, D0):
        return cv.calibrateCameraExtended(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=(W, H),
            cameraMatrix=K0.copy(),
            distCoeffs=D0.copy(),
            flags=flags
        )

    # ---------- Калибровка: один или два прохода ----------
    if args.two_pass:
        flags1 = (flags_base & ~(cv.CALIB_THIN_PRISM_MODEL | cv.CALIB_TILTED_MODEL))
        if args.fix_aspect:
            flags1 |= cv.CALIB_FIX_ASPECT_RATIO
            f = float((initK[0,0] + initK[1,1]) * 0.5)
            initK[0,0] = f
            initK[1,1] = f
        rms1, K1, D1, *_ = _calib(flags1, initK, np.zeros((14,1), np.float64))
        print(f"[calib] pass1 RMS: {rms1:.4f}")
        rms, K, dist, rvecs, tvecs, std_i, std_e, per_view = _calib(flags_base, K1, D1)
    else:
        rms, K, dist, rvecs, tvecs, std_i, std_e, per_view = _calib(flags_base, initK, np.zeros((14,1), np.float64))

    print(f"[calib] RMS reprojection error: {rms:.4f}")
    print(f"[calib] K:\n{K}\n[calib] dist shape: {dist.shape}")
    fovx = 2*np.degrees(np.arctan(W/(2*float(K[0,0]))))
    fovy = 2*np.degrees(np.arctan(H/(2*float(K[1,1]))))
    print(f"[calib] FOVx≈{fovx:.1f}°, FOVy≈{fovy:.1f}°")

    # ---------- Оптимальные «камеры вывода» ----------
    newK0, _ = cv.getOptimalNewCameraMatrix(K, dist, (W,H), alpha=0.0, newImgSize=(W,H), centerPrincipalPoint=False)
    newK1, _ = cv.getOptimalNewCameraMatrix(K, dist, (W,H), alpha=1.0, newImgSize=(W,H), centerPrincipalPoint=False)

    # ---------- Быстрый предпросмотр undistort ----------
    if not args.no_preview:
        mapx, mapy = cv.initUndistortRectifyMap(K, dist, np.eye(3), newK0, (W,H), cv.CV_32FC1)
        und = cv.remap(frames[0], mapx, mapy, cv.INTER_LINEAR)
        imshow_resized("preview_undistort_alpha0", und, maxw=args.maxw, maxh=args.maxh)
        cv.waitKey(400)
        cv.destroyAllWindows()

    # ---------- Сохранение ----------
    np.savez(args.out,
        K=K, dist=dist, image_size=np.array([W,H], np.int32),
        model=("pinhole_" + args.model),
        flags=np.int32(flags_base), rms=np.float64(rms),
        newK_alpha0=newK0, newK_alpha1=newK1
    )
    print(f"[calib] Сохранено: {args.out}")

if __name__ == "__main__":
    main()
