# stereo_rtsp_runtime.py
import os, time, argparse
from pathlib import Path
import cv2
import numpy as np
from collections import deque

class StereoMotionDetector:
    def __init__(self, w, h,
                 y_eps=8, dmin=2, dmax=400,           # стерео-ґейтинг (px)
                 min_area=30, max_area=2000,           # фільтр площі (px^2)
                 thr_fast=14, thr_slow=6,              # пороги для швидкого/повільного фону
                 alpha_fast=0.40, alpha_slow=0.02,     # коеф. навчаня фонів
                 min_flow=0.6):                        # мін. швидкість (px/frame)
        self.size = (w, h)
        self.y_eps, self.dmin, self.dmax = y_eps, dmin, dmax
        self.min_area, self.max_area = min_area, max_area
        self.thr_fast, self.thr_slow = thr_fast, thr_slow
        self.alpha_fast, self.alpha_slow = alpha_fast, alpha_slow
        self.min_flow = min_flow

        self.bg_fast_L = None; self.bg_slow_L = None
        self.bg_fast_R = None; self.bg_slow_R = None
        self.prevL = None; self.prevR = None
        self.k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        # для згладжування детекцій (3 останні кадри)
        self.masksL = deque(maxlen=3)
        self.masksR = deque(maxlen=3)

    def _prep(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # легка нормалізація яскравості проти флікера
        g = cv2.equalizeHist(g)
        return g

    def _update_bg(self, g, which):
        bg_fast = getattr(self, f'bg_fast_{which}')
        bg_slow = getattr(self, f'bg_slow_{which}')
        if bg_fast is None:
            setattr(self, f'bg_fast_{which}', g.astype(np.float32))
            setattr(self, f'bg_slow_{which}', g.astype(np.float32))
            return np.zeros_like(g)

        cv2.accumulateWeighted(g, getattr(self, f'bg_fast_{which}'), self.alpha_fast)
        cv2.accumulateWeighted(g, getattr(self, f'bg_slow_{which}'), self.alpha_slow)

        fast = cv2.convertScaleAbs(cv2.absdiff(g, getattr(self, f'bg_fast_{which}')))
        slow = cv2.convertScaleAbs(cv2.absdiff(g, getattr(self, f'bg_slow_{which}')))
        # “подвійне” віднімання: швидке – повільне
        resp = cv2.subtract(fast, slow)
        _, m = cv2.threshold(resp, self.thr_fast, 255, cv2.THRESH_BINARY)
        # прибрати залишки повільних змін
        m[slow > self.thr_slow] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k3, iterations=1)
        m = cv2.dilate(m, self.k3, iterations=1)
        return m

    def _boxes(self, m):
        cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out=[]
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.min_area or a > self.max_area: continue
            x,y,w,h = cv2.boundingRect(c)
            out.append((x,y,w,h))
        return out

    def _flow_mag(self, prev, cur):
        # швидкий флоу на зменшеній копії
        small_prev = cv2.resize(prev, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        small_cur  = cv2.resize(cur,  (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(small_prev, small_cur, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag = cv2.magnitude(flow[...,0], flow[...,1]) * 2.0  # масштаб назад
        return cv2.resize(mag, self.size, interpolation=cv2.INTER_LINEAR)

    def step(self, rL, rR):
        # підготовка
        gL, gR = self._prep(rL), self._prep(rR)

        # маски подвійного фону
        mL = self._update_bg(gL, 'L')
        mR = self._update_bg(gR, 'R')

        # легке темпоральне згладжування масок
        self.masksL.append(mL); self.masksR.append(mR)
        mL = cv2.bitwise_and.reduce(self.masksL) if len(self.masksL)==self.masksL.maxlen else mL
        mR = cv2.bitwise_and.reduce(self.masksR) if len(self.masksR)==self.masksR.maxlen else mR

        boxesL = self._boxes(mL)
        boxesR = self._boxes(mR)

        # швидкісний фільтр (прибирає “димно-хмарні” плями)
        if self.prevL is not None and self.prevR is not None:
            magL = self._flow_mag(self.prevL, gL)
            magR = self._flow_mag(self.prevR, gR)
            boxesL = [b for b in boxesL if np.median(magL[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]) > self.min_flow]
            boxesR = [b for b in boxesR if np.median(magR[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]) > self.min_flow]

        self.prevL, self.prevR = gL, gR

        # СТЕРЕО-ґейтінг: y близький, диспаритет у [dmin,dmax]
        matches=[]
        for (xL,yL,wL,hL) in boxesL:
            cyL = yL + hL//2
            for (xR,yR,wR,hR) in boxesR:
                cyR = yR + hR//2
                if abs(cyL - cyR) > self.y_eps: continue
                disp = (xL + wL//2) - (xR + wR//2)
                if self.dmin <= abs(disp) <= self.dmax:
                    matches.append(((xL,yL,wL,hL),(xR,yR,wR,hR)))
                    break  # у простому варіанті — перша валідна пара
        return mL, mR, boxesL, boxesR, matches


def match_rectified_gated(boxesL, boxesR, y_eps=8, dmin=4, dmax=400):
    """Повертає [(bL, bR, disp)] для ректифікованих кадрів.
       Унікально підбирає праву коробку під кожну ліву."""
    matches = []
    usedR = set()
    for bL in boxesL:
        xL,yL,wL,hL = bL
        cL = (xL + wL//2, yL + hL//2)
        best = None; bestj = None
        for j,bR in enumerate(boxesR):
            if j in usedR: continue
            xR,yR,wR,hR = bR
            cR = (xR + wR//2, yR + hR//2)
            if abs(cL[1] - cR[1]) > y_eps:
                continue
            disp = cL[0] - cR[0]
            if dmin <= abs(disp) <= dmax:
                # беремо найближчий по Y
                dy = abs(cL[1] - cR[1])
                if best is None or dy < best[0]:
                    best = (dy, (bL, bR, disp)); bestj = j
        if best is not None:
            matches.append(best[1]); usedR.add(bestj)
    return matches

def Rz_from_deg(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)

# ---------- IO / RTSP ----------

def set_ffmpeg_tcp():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|rtsp_flags;prefer_tcp|"
        "allowed_media_types;video|"        # ігнорувати аудіо
        "stimeout;7000000|max_delay;7000000|"
        "buffer_size;2097152|"
        "analyzeduration;2000000|probesize;2000000|"
        "loglevel;error"                    # менше шуму в консолі
    )

def open_rtsp(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# ---------- Intrinsics loaders & helpers ----------

def load_intrinsics_npz(npz_path):
    """Читає intrinsics.npz з ключами K / dist (+ image_size якщо є)."""
    data = np.load(npz_path, allow_pickle=False)
    K = None; D = None
    for k in ("K","camera_matrix","K1","mtx"):
        if k in data: K = np.asarray(data[k], np.float64)
    for k in ("dist","D","distCoeffs","distortion","D1"):
        if k in data: D = np.asarray(data[k], np.float64)
    if K is None:
        raise ValueError(f"{npz_path}: немає K (очікував K/camera_matrix/K1/mtx)")
    if D is None:
        D = np.zeros((5,), np.float64)
    if D.ndim > 1:
        D = D.reshape(-1)

    calib_size = None
    if "image_size" in data:
        w, h = map(int, data["image_size"])  # (w, h)
        calib_size = (w, h)
    return K, D, calib_size

def scale_intrinsics(K, old_wh, new_wh):
    """Масштабує fx,fy,cx,cy з old_wh -> new_wh."""
    old_w, old_h = old_wh; new_w, new_h = new_wh
    sx = new_w / float(old_w); sy = new_h / float(old_h)
    K = K.copy().astype(np.float64)
    K[0,0] *= sx;  K[1,1] *= sy
    K[0,2] *= sx;  K[1,2] *= sy
    return K

# (опціонально для відладки H)
def sanity_check_maps(mapx, mapy, w, h, name="map"):
    print(f"[DEBUG] {name}: shape={mapx.shape}, dtype={mapx.dtype}")
    print(f"[DEBUG] {name} ranges: x[{float(mapx.min()):.2f}, {float(mapx.max()):.2f}], "
          f"y[{float(mapy.min()):.2f}, {float(mapy.max()):.2f}]")
    pts = [(0,0), (w-1,0), (w-1,h-1), (0,h-1), (w//2, h//2)]
    for (u,v) in pts:
        sx = float(mapx[v,u]); sy = float(mapy[v,u])
        print(f"[DEBUG] {name} dst({u},{v}) -> src({sx:.1f},{sy:.1f})")

# ---------- Calibration loader ----------

def load_calibration(calib_dir, intrinsics_npz=None, baseline_m=None, frame_size=None,
                     rect_alpha=1.0, ignore_dist=False, keep_P_as_K=False,
                     debug=False, autoroll=False, roll_deg=0.0):
    """
    Повертає dict:
      mode: "metric" | "projective"
      maps: (map1x,map1y,map2x,map2y)  # якщо metric
      H:    (H1,H2)                    # якщо projective
      Q:    Q або None
      P:    (P1,P2) або (None,None)
    """
    d = Path(calib_dir)
    if not d.exists():
        raise FileNotFoundError(calib_dir)

    # Метрична (рекомендована): R_avg + t_dir_avg + intrinsics
    Rf, tf = d/"R_avg.npy", d/"t_dir_avg.npy"
    if intrinsics_npz and Rf.exists() and tf.exists():
        if frame_size is None:
            raise RuntimeError("Передай --width/--height (frame_size) для побудови карт")

        K, D, npz_size = load_intrinsics_npz(intrinsics_npz)
        if npz_size is not None and npz_size != frame_size:
            K = scale_intrinsics(K, npz_size, frame_size)

        R_avg = np.load(Rf)
        t_dir = np.load(tf).reshape(3,)

        if baseline_m is None:
            print("[!] --baseline не заданий: Z буде в умовних одиницях")
            baseline_m = 1.0

        # Можлива підміна дисторсії
        D_used = np.zeros((5,), np.float64) if ignore_dist else D

        w, h = frame_size
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K, D_used, K, D_used, (w, h),
            R_avg, (t_dir * baseline_m).reshape(3,1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=rect_alpha
        )
        if keep_P_as_K:
            P1 = K.copy()
            P2 = K.copy()

        if autoroll or abs(roll_deg) > 1e-9:
            # оцінюємо вбудований рол (кут у площині) з R1
            # (кут, на який повернута вісь x відносно вихідного зображення)
            auto_deg = np.degrees(np.arctan2(R1[0,1], R1[0,0])) if autoroll else 0.0
            total_deg = -auto_deg + roll_deg   # мінус, щоб занулити, + ручний зсув
            Rz = Rz_from_deg(total_deg)

            # застосовуємо однакову оберталку до обох ректифікацій
            R1 = Rz @ R1
            R2 = Rz @ R2

        map1x, map1y = cv2.initUndistortRectifyMap(K, D_used, R1, P1, (w,h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K, D_used, R2, P2, (w,h), cv2.CV_32FC1)

        if debug:
            sanity_check_maps(map1x, map1y, w, h, "map1")
            sanity_check_maps(map2x, map2y, w, h, "map2")

        return dict(mode="metric", maps=(map1x,map1y,map2x,map2y), H=None, Q=Q, P=(P1,P2))

    # Фолбек: якщо є лише H1/H2 (проєктна, без абсолютної глибини)
    H1p, H2p = d/"H1.npy", d/"H2.npy"
    if H1p.exists() and H2p.exists():
        H1 = np.load(H1p).astype(np.float64).reshape(3,3)
        H2 = np.load(H2p).astype(np.float64).reshape(3,3)
        if abs(H1[2,2]) > 1e-12: H1 /= H1[2,2]
        if abs(H2[2,2]) > 1e-12: H2 /= H2[2,2]
        return dict(mode="projective", maps=None, H=(H1,H2), Q=None, P=(None,None))

    raise RuntimeError("У теці калібровки немає ні метричного (R_avg/t_dir_avg + intrinsics), ні проектного (H1/H2) результату")

# ---------- Simple motion detection & helpers ----------

def make_motion_detector():
    bsL = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=32, detectShadows=True)
    bsR = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=32, detectShadows=True)
    K = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return bsL, bsR, K

def motion_bboxes(frame, bs, K,
                  min_area=30, max_area=2000,
                  thr_fast=14, thr_slow=6,
                  alpha_fast=0.40, alpha_slow=0.02,
                  min_flow=0.7):
    import numpy as np
    import cv2

    # --- локальний стан для кожної камери (ключимо по id(bs)) ---
    if not hasattr(motion_bboxes, "_st"):
        motion_bboxes._st = {}
    st = motion_bboxes._st.setdefault(id(bs), {
        "bg_fast": None, "bg_slow": None, "prev": None
    })

    # підготовка + стабілізація яскравості
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g32 = g.astype(np.float32)  # <-- однаковий тип для diff та accumulateWeighted

    # ініціалізація фонів
    if st["bg_fast"] is None:
        st["bg_fast"] = g32.copy()
        st["bg_slow"] = g32.copy()
        st["prev"]    = g.copy()
        return [], np.zeros_like(g, dtype=np.uint8)

    # оновлення фонів (працюємо у float32)
    cv2.accumulateWeighted(g32, st["bg_fast"], alpha_fast)
    cv2.accumulateWeighted(g32, st["bg_slow"], alpha_slow)

    # різниця теж у float32, потім конвертуємо в u8
    fast_f = cv2.absdiff(g32, st["bg_fast"])
    slow_f = cv2.absdiff(g32, st["bg_slow"])
    fast   = cv2.convertScaleAbs(fast_f)
    slow   = cv2.convertScaleAbs(slow_f)

    resp = cv2.subtract(fast, slow)
    _, m = cv2.threshold(resp, thr_fast, 255, cv2.THRESH_BINARY)
    m[slow > thr_slow] = 0

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K, iterations=1)
    m = cv2.dilate(m, K, iterations=1)

    # контури за площею
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))

    # швидкісний фільтр (оптичний флоу), відсікає “димні/хмарні” плями
    if st["prev"] is not None and boxes:
        # рахуємо флоу на півмасштабі для швидкості
        p0 = cv2.resize(st["prev"], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        p1 = cv2.resize(g,          (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(p0, p1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = cv2.magnitude(flow[...,0], flow[...,1]) * 2.0  # повертаємо масштаб
        mag = cv2.resize(mag, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)

        boxes = [b for b in boxes
                 if np.median(mag[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]) > min_flow]

    st["prev"] = g
    return boxes, m

def match_rectified(boxesL, boxesR, y_eps=8):
    matches=[]
    for (xL,yL,wL,hL) in boxesL:
        cyL = yL + hL//2
        for (xR,yR,wR,hR) in boxesR:
            cyR = yR + hR//2
            if abs(cyL - cyR) <= y_eps:
                matches.append(((xL,yL,wL,hL),(xR,yR,wR,hR)))
    return matches

def reproject_point_Q(xL, y, xR, Q):
    d = float(xL - xR)
    X = Q @ np.array([xL, y, d, 1.0], dtype=np.float64)
    X /= X[3] + 1e-12
    return X[:3]

def show_resized(win, img, max_w=1280, max_h=720):
    h,w = img.shape[:2]
    s = min(max_w/w, max_h/h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win, img)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser("Рантайм з використанням калібровки (дві RTSP камери)")
    ap.add_argument("--left",  required=True, help="RTSP лівої камери")
    ap.add_argument("--right", required=True, help="RTSP правої камери")
    ap.add_argument("--calib_dir", required=True, help="Тека з результатами калібровки")
    ap.add_argument("--intrinsics", help="intrinsics.npz (K, dist, image_size) – для метричного режиму")
    ap.add_argument("--baseline", type=float, help="база між камерами в метрах")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--y_eps", type=int, default=8, help="допуск по Y для матчінгу")
    ap.add_argument("--rect_alpha", type=float, default=1.0, help="0..1: частка FOV при stereoRectify (1.0 = максимум)")
    ap.add_argument("--ignore_dist", action="store_true", help="ігнорувати дисторсію з intrinsics (використати D=0)")
    ap.add_argument("--keep_P_as_K", action="store_true", help="залишити P1=P2=K (не змінювати фокус)")
    ap.add_argument("--debug", action="store_true", help="друкувати діапазони карт і розміри")
    ap.add_argument("--autoroll", action="store_true",
                help="автоматично занулити спільний рол (за R1)")
    ap.add_argument("--roll_deg", type=float, default=0.0,
                help="додатково повернути на цей кут (°), +CW / -CCW у підсумку")
    ap.add_argument("--dmin", type=int, default=4,   help="мін. модуль диспаритету (px) — відсікає «нескінченні» хмари/блиски")
    ap.add_argument("--dmax", type=int, default=400, help="макс. модуль диспаритету (px) — відсікає зовсім близькі артефакти")
    args = ap.parse_args()

    set_ffmpeg_tcp()
    capL, capR = open_rtsp(args.left), open_rtsp(args.right)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Не відкрився один із RTSP потоків")

    # Перевіримо реальні розміри RTSP
    okL, tL = capL.read(); okR, tR = capR.read()
    if okL and okR:
        print(f"RTSP real sizes: {tL.shape[1]} x {tL.shape[0]} | {tR.shape[1]} x {tR.shape[0]}")
    else:
        print("[!] Не вдалось прочитати тестові кадри")

    calib = load_calibration(
        args.calib_dir,
        intrinsics_npz=args.intrinsics,
        baseline_m=args.baseline,
        frame_size=(args.width, args.height),
        rect_alpha=args.rect_alpha,
        ignore_dist=args.ignore_dist,
        keep_P_as_K=args.keep_P_as_K,
        debug=args.debug,
        autoroll=args.autoroll,
        roll_deg=args.roll_deg
    )
    mode = calib["mode"]
    print(f"[*] Режим калібровки: {mode}")

    if mode == "metric":
        map1x,map1y,map2x,map2y = calib["maps"]
        Q = calib["Q"]
    else:
        H1,H2 = calib["H"]
        Q = None

    bsL, bsR, Kker = make_motion_detector()

    bad = 0
    last_ok = time.time()

    while True:
        capL.grab(); capR.grab()
        okL, fL = capL.retrieve(); okR, fR = capR.retrieve()

        if not (okL and okR):
            bad += 1
            if bad > 30 or (time.time() - last_ok) > 5:
                capL.release(); capR.release()
                time.sleep(0.5)
                capL, capR = open_rtsp(args.left), open_rtsp(args.right)
                bad = 0
            continue

        bad = 0
        last_ok = time.time()

        # привести до поточного розміру
        if (fL.shape[1], fL.shape[0]) != (args.width, args.height):
            fL = cv2.resize(fL, (args.width, args.height), interpolation=cv2.INTER_AREA)
        if (fR.shape[1], fR.shape[0]) != (args.width, args.height):
            fR = cv2.resize(fR, (args.width, args.height), interpolation=cv2.INTER_AREA)

        # ректифікація
        if mode == "metric":
            rL = cv2.remap(fL, map1x, map1y, cv2.INTER_LINEAR)
            rR = cv2.remap(fR, map2x, map2y, cv2.INTER_LINEAR)
        else:
            rL = cv2.warpPerspective(fL, H1, (args.width, args.height))
            rR = cv2.warpPerspective(fR, H2, (args.width, args.height))

        # проста детекція (можна прибрати/замінити своїм кодом)
        gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
        boxesL, mL = motion_bboxes(rL, bsL, Kker, min_area=20, max_area=1500)
        boxesR, mR = motion_bboxes(rR, bsR, Kker, min_area=20, max_area=1500)

        outL = rL.copy(); outR = rR.copy()

        # зелені рамки — «сирі» детекції з кожної камери
        for (x,y,w,h) in boxesL: cv2.rectangle(outL,(x,y),(x+w,y+h),(0,255,0),1)
        for (x,y,w,h) in boxesR: cv2.rectangle(outR,(x,y),(x+w,y+h),(0,255,0),1)

        # пари з ґейтінгом по диспаритету
        pairs = match_rectified_gated(boxesL, boxesR, y_eps=args.y_eps,
                                    dmin=args.dmin, dmax=args.dmax)

        for (bL, bR, disp) in pairs:
            (xL,yL,wL,hL) = bL; (xR,yR,wR,hR) = bR
            cL = (xL + wL//2, yL + hL//2)
            cR = (xR + wR//2, yR + hR//2)

            # червоні рамки — підтверджені стерео-об’єкти
            cv2.rectangle(outL,(xL,yL),(xL+wL,yL+hL),(0,0,255),2)
            cv2.rectangle(outR,(xR,yR),(xR+wR,yR+hR),(0,0,255),2)
            cv2.circle(outL, cL, 3, (0,0,255), -1)
            cv2.circle(outR, cR, 3, (0,0,255), -1)

            label = f"d={disp:.1f}px"
            if Q is not None:
                X,Y,Z = reproject_point_Q(float(cL[0]), float(cL[1]), float(cR[0]), Q)
                label += f"  Z={Z:.1f}m"
            cv2.putText(outL, label, (xL, max(15, yL-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # показ
        show_resized("Left rect",  outL, 1280, 720)
        show_resized("Right rect", outR, 1280, 720)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capL.release(); capR.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
