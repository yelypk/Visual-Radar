# stereo_rtsp_runtime.py
import os, time, argparse, json
from pathlib import Path
from collections import deque
import subprocess, numpy as np, cv2, time, shutil
import threading, queue

os.environ['OPENCV_OPENCL_RUNTIME'] = 'disabled'
cv2.ocl.setUseOpenCL(False)

class FFmpegRTSP_MJPEG:
    def __init__(self, url, width, height, use_tcp=True, ffmpeg_cmd="ffmpeg", q=6, threads=3, read_timeout=0.8):
        self.ffmpeg_cmd = ffmpeg_cmd
        self.w, self.h = int(width), int(height)
        self.url = url
        self.use_tcp = use_tcp
        self.q = int(q)
        self.threads = int(threads)
        self.read_timeout = float(read_timeout)
        self.proc = None
        self.buf = bytearray()
        self._q = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thr = None
        self._start()

    def _start(self):
        if not (shutil.which(self.ffmpeg_cmd) or os.path.exists(self.ffmpeg_cmd)):
            raise RuntimeError(f"ffmpeg не знайдено: {self.ffmpeg_cmd}")
        cmd = [
            self.ffmpeg_cmd,
            "-rtsp_transport", "tcp" if self.use_tcp else "udp",
            "-rtsp_flags", "prefer_tcp",
            
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-hide_banner", "-loglevel", "warning",
            "-fflags", "nobuffer+discardcorrupt+genpts",
            "-flags", "+low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-max_delay", "500000",
            "-i", self.url,
            "-an",
            "-vf", f"scale={self.w}:{self.h}:flags=bicubic",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-threads", str(self.threads),
            "-q:v", str(self.q),
            "-"
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=None, bufsize=10**7
        )
        self.buf.clear()
        self._stop.clear()
        self._thr = threading.Thread(target=self._reader, daemon=True)
        self._thr.start()

    def _reader(self):
        CHUNK = 65536
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        while not self._stop.is_set() and self.proc and self.proc.poll() is None:
            try:
                data = self.proc.stdout.read(CHUNK)
                if not data:
                    time.sleep(0.01)
                    continue
                self.buf += data
                while True:
                    i = self.buf.find(SOI)
                    if i == -1:
                        if len(self.buf) > 10*CHUNK:
                            self.buf.clear()
                        break
                    j = self.buf.find(EOI, i+2)
                    if j == -1:
                        if len(self.buf) > 20*CHUNK:
                            del self.buf[:i]
                        break
                    jpg = bytes(self.buf[i:j+2])
                    del self.buf[:j+2]
                    try:
                        self._q.put_nowait(jpg)
                    except queue.Full:
                        pass
            except Exception:
                time.sleep(0.01)
                continue

    # --- API сумісність з OpenCV ---
    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def release(self):
        try:
            self._stop.set()
            if self._thr and self._thr.is_alive():
                self._thr.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.proc:
                self.proc.kill(); self.proc.stdout.close()
        except Exception:
            pass
        self.proc = None
        self._thr = None
        with self._q.mutex:
            self._q.queue.clear()

    def grab(self):
        return True

    def retrieve(self):
        if self.proc is None or self.proc.poll() is not None:
            self.release(); self._start(); time.sleep(0.1)
        try:
            jpg = self._q.get(timeout=self.read_timeout)
        except queue.Empty:
            return False, None
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False, None
        if (img.shape[1], img.shape[0]) != (self.w, self.h):
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return True, img

    def read(self):
        return self.retrieve()

# ---------- Motion detector (подвійний фон + флоу) ----------
class StereoMotionDetector:
    def __init__(self, size, y_eps=8, dmin=4, dmax=400,
                 min_flow_small=0.01, stereo_patch=13,
                 stereo_ncc_min=0.50, stereo_search_pad=40):
        self.size = size
        self.w, self.h = size
        self.k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.alpha_fast = 0.40
        self.alpha_slow = 0.02
        self.thr_fast = 14
        self.thr_slow = 6
        self.min_area = 30
        self.max_area = 2000
        self.min_flow  = 0.7
        self.y_eps = y_eps
        self.dmin = dmin
        self.dmax = dmax
        # параметри епіполярного NCC-fallback — NEW
        self.stereo_patch = int(stereo_patch)
        self.stereo_ncc_min = float(stereo_ncc_min)
        self.stereo_search_pad = int(stereo_search_pad)
        # окремі стани для лівої/правої
        self.bg_fast_L = None; self.bg_slow_L = None
        self.bg_fast_R = None; self.bg_slow_R = None
        self.prevL = None; self.prevR = None
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
            if a < self.min_area or a > self.max_area:
                continue
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

        # оновити фони
        mL = self._update_bg(gL, 'L')
        mR = self._update_bg(gR, 'R')

        # легке темпоральне згладжування масок
        self.masksL.append(mL); self.masksR.append(mR)
        mL = np.bitwise_and.reduce(self.masksL) if len(self.masksL)==self.masksL.maxlen else mL
        mR = np.bitwise_and.reduce(self.masksR) if len(self.masksR)==self.masksR.maxlen else mR

        boxesL = self._boxes(mL)
        boxesR = self._boxes(mR)

        # швидкісний фільтр (прибирає “димно-хмарні” плями)
        if self.prevL is not None and self.prevR is not None:
            magL = self._flow_mag(self.prevL, gL)
            magR = self._flow_mag(self.prevR, gR)
            def pass_flow(box, mag):
                x,y,w,h = box
                m = np.median(mag[y:y+h, x:x+w])
                # для дуже дрібних боксів дозволяємо повільніший рух
                thr = self.min_flow_small if min(w,h) <= 10 else self.min_flow
                return m > thr
            boxesL = [b for b in boxesL if pass_flow(b, magL)]
            boxesR = [b for b in boxesR if pass_flow(b, magR)]

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
        # Якщо для лівого боксу не знайшлося пари — спробуємо NCC-відбір по епіполярі
        if not matches:
            gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
            for (xL,yL,wL,hL) in boxesL:
                cLx, cLy = xL + wL//2, yL + hL//2
                xR_best, ncc = epipolar_ncc_match(
                    gL, gR, (cLx, cLy), self.dmin, self.dmax,
                    patch=self.stereo_patch,
                    ncc_min=self.stereo_ncc_min,
                    search_pad=self.stereo_search_pad
                )
                if xR_best is None:
                    continue
                disp = (cLx - xR_best)
                # синтезуємо правий бокс такої ж ширини/висоти, центр в (xR_best,cLy)
                wR, hR = wL, hL
                xR = max(0, min(int(xR_best - wR//2), rR.shape[1]-wR))
                yR = max(0, min(int(cLy - hR//2),  rR.shape[0]-hR))
                matches.append(((xL,yL,wL,hL),(xR,yR,wR,hR)))

        return mL, mR, boxesL, boxesR, matches


def match_rectified_gated(boxesL, boxesR, y_eps=8, dmin=4, dmax=400):
    """Повертає [(bL, bR, disp)] для ректифікованих кадрів.
       Унікально підбирає праву коробку під кожну ліву."""
    matches = []
    usedR = set()
    for i,bL in enumerate(boxesL):
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
                dy = abs(cL[1] - cR[1])
                if best is None or dy < best[0]:
                    best = (dy, (bL, bR, disp)); bestj = j
        if best is not None:
            matches.append(best[1]); usedR.add(bestj)
    return matches

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
    sx = new_wh[0] / float(old_wh[0])
    sy = new_wh[1] / float(old_wh[1])
    S = np.diag([sx, sy, 1.0])
    K2 = K.copy().astype(np.float64)
    K2[:2,:] = S @ K2[:2,:]
    return K2


def load_calibration(calib_dir, intrinsics_npz=None, baseline_m=None,
                     rect_alpha=0.0, ignore_dist=False, keep_P_as_K=False,
                     frame_size=None, autoroll=False, roll_deg=0.0):
    """Завантажити калібровку і побудувати карти ректифікації.

    calib_dir очікує файли:
      - F_avg.npy          — середня фундаментальна матриця
      - H1.npy / H2.npy    — ректифікації (uncalibrated) — опційно
      - R_from_Rt.npy      — середня R (від стерео з ландшафту)
      - t_dir.npy          — одиничний напрямок базису t (нормований), без масштабу
    Якщо intrinsics_npz заданий — будуємо «metric rectification» через stereoRectify.
    """
    calib_dir = Path(calib_dir)
    Fp = calib_dir/"F_avg.npy"
    H1p = calib_dir/"H1.npy"; H2p = calib_dir/"H2.npy"

    # Підтримка кількох назв файлів (як у твоїй папці)
    def _pick(*names):
        for n in names:
            p = calib_dir / n
            if p.exists():
                return p
        raise FileNotFoundError(f"Не знайдено жодного з: {', '.join(names)} в {calib_dir}")

    # R, t можуть називатись по-різному: R_from_Rt.npy / R_avg.npy, t_dir.npy / t_dir_avg.npy
    Rf = _pick("R_from_Rt.npy", "R_avg.npy", "R.npy")
    tf = _pick("t_dir.npy", "t_dir_avg.npy", "t.npy")

    if not Fp.exists():
        raise FileNotFoundError(f"{Fp} не знайдено — спочатку запусти калібрування")

    have_metric = intrinsics_npz is not None and Path(intrinsics_npz).exists()

    if not have_metric:
        # використовуємо не-метричну ректифікацію
        if not (H1p.exists() and H2p.exists()):
            raise FileNotFoundError("Немає H1/H2 для проектної ректифікації")
        H1 = np.load(H1p); H2 = np.load(H2p)
        print("[*] Режим калібровки: projective (uncalibrated)")
        return {"mode":"proj","H":(H1,H2),"Q":None}

    # --- metric rectification ---
    # потрібні K,D і базова геометрія R,t
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
        roll = auto_deg + roll_deg
        if abs(roll) > 1e-6:
            c = np.cos(np.radians(roll)); s = np.sin(np.radians(roll))
            Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float64)
            R1 = Rz @ R1; R2 = Rz @ R2

    # карти
    map1x, map1y = cv2.initUndistortRectifyMap(K, D_used, R1, P1, (w,h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, D_used, R2, P2, (w,h), cv2.CV_32FC1)

    print("[*] Режим калібровки: metric")
    return {"mode":"metric","maps":(map1x,map1y,map2x,map2y),"Q":Q}


def set_ffmpeg_tcp():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|rtsp_flags;prefer_tcp|"
        "allowed_media_types;video|"
        "max_delay;7000000|"
        "buffer_size;4194304|"
        "analyzeduration;2000000|probesize;2000000|"
        "loglevel;error"
    )

# ---------- старий простий детектор (залишено для сумісності/порівняння) ----------
def make_motion_detector(kernel=3):
    bsL = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)
    bsR = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel), int(kernel)))
    return bsL, bsR, K


def motion_bboxes(frame, bs, K,
                  min_area=30, max_area=2000,
                  thr_fast=14, thr_slow=6,
                  alpha_fast=0.40, alpha_slow=0.02,
                  min_flow=0.7,
                  temporal='none'):
    import numpy as np
    import cv2

    # --- локальний стан для кожної камери (ключимо по id(bs)) ---
    if not hasattr(motion_bboxes, "_st"):
        motion_bboxes._st = {}
    st = motion_bboxes._st.setdefault(id(bs), {
        "bg_fast": None, "bg_slow": None, "prev": None, "masks": None
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

    fast_f = cv2.absdiff(g32, st["bg_fast"]) ; slow_f = cv2.absdiff(g32, st["bg_slow"])
    fast   = cv2.convertScaleAbs(fast_f)
    slow   = cv2.convertScaleAbs(slow_f)

    resp = cv2.subtract(fast, slow)
    _, m = cv2.threshold(resp, thr_fast, 255, cv2.THRESH_BINARY)
    m[slow > thr_slow] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K, iterations=1)
    m = cv2.dilate(m, K, iterations=1)

    # темпоральне згладжування: none / and3 / maj3 (2 з 3)
    if temporal != 'none':
        from collections import deque
        if st['masks'] is None:
            st['masks'] = deque(maxlen=3)
        st['masks'].append(m.copy())
        if len(st['masks']) == 3:
            a,b,c = st['masks']
            if temporal == 'and3':
                m = cv2.bitwise_and(cv2.bitwise_and(a,b), c)
            elif temporal == 'maj3':
                m = cv2.bitwise_or(cv2.bitwise_and(a,b), cv2.bitwise_or(cv2.bitwise_and(b,c), cv2.bitwise_and(a,c)))

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area: continue
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,w,h))

    # фільтр швидкості
    if st["prev"] is not None and boxes:
        p0 = cv2.resize(st["prev"], (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        p1 = cv2.resize(g,           (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(p0, p1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = cv2.magnitude(flow[...,0], flow[...,1]) * 2.0
        mag = cv2.resize(mag, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        boxes = [b for b in boxes if np.median(mag[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]) > min_flow]

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

def _ncc(a, b):
    # коеф. кореляції (NCC) через matchTemplate; обидві латки в градаціях сірого
    a = a.astype(np.uint8); b = b.astype(np.uint8)
    if a.shape != b.shape or a.size < 36:
        return 0.0
    res = cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)
    return float(res[0,0])

class SnapshotSaver:
    def __init__(self, out_dir, min_disp=1.5, min_cc=0.60,
                 zmin=3.0, zmax=5000.0, cooldown=1.5, pad=4, debug=False):
        self.out_dir = Path(out_dir)
        self.min_disp = float(min_disp)
        self.min_cc   = float(min_cc)
        self.zmin, self.zmax = float(zmin), float(zmax)
        self.cooldown = float(cooldown)
        self.pad = int(pad)
        self.debug = bool(debug)
        self._recent = []  # (t, cx, cy)

    def _too_close_recent(self, cx, cy, now):
        # антидублювання поблизу того ж місця упродовж cooldown
        self._recent = [(t,x,y) for (t,x,y) in self._recent if (now - t) < self.cooldown]
        for (t,x,y) in self._recent:
            if abs(cx-x) <= 24 and abs(cy-y) <= 24:
                return True
        return False

    def maybe_save(self, rL, rR, boxL, boxR, disp, Z, now=None):
        # 1) базові перевірки «впевненості»
        if Z is None or not np.isfinite(Z):
            if self.debug: print("[snap:skip] invalid Z")
            return False
        if abs(disp) < self.min_disp:
            if self.debug: print(f"[snap:skip] disp {disp:.2f} < {self.min_disp}")
            return False
        if not (self.zmin <= Z <= self.zmax):
            if self.debug: print(f"[snap:skip] Z {Z:.1f} not in [{self.zmin},{self.zmax}]")
            return False

        xL,yL,wL,hL = boxL; xR,yR,wR,hR = boxR
        cx, cy = xL + wL//2, yL + hL//2
        now = now or time.time()
        if self._too_close_recent(cx, cy, now):
            if self.debug: print("[snap:skip] cooldown")
            return False

        # 2) вирізати однакові латки навколо центрів
        half = max(8, min(wL,hL,wR,hR)//2) + self.pad
        y1L, y2L = max(0, cy-half), min(rL.shape[0], cy+half)
        x1L, x2L = max(0, cx-half), min(rL.shape[1], cx+half)
        cR_x, cR_y = xR + wR//2, yR + hR//2
        y1R, y2R = max(0, cR_y-half), min(rR.shape[0], cR_y+half)
        x1R, x2R = max(0, cR_x-half), min(rR.shape[1], cR_x+half)

        cropL = rL[y1L:y2L, x1L:x2L]
        cropR = rR[y1R:y2R, x1R:x2R]
        if cropL.size == 0 or cropR.size == 0 or cropL.shape[:2] != cropR.shape[:2]:
            if self.debug:
                print(f"[snap:skip] crop mismatch L{getattr(cropL,'shape',None)} R{getattr(cropR,'shape',None)}")
            return False

        grayL = cv2.cvtColor(cropL, cv2.COLOR_BGR2GRAY) if cropL.ndim==3 else cropL
        grayR = cv2.cvtColor(cropR, cv2.COLOR_BGR2GRAY) if cropR.ndim==3 else cropR
        cc = _ncc(grayL, grayR)
        if cc < self.min_cc:
            if self.debug: print(f"[snap:skip] NCC {cc:.2f} < {self.min_cc}")
            return False

        # 3) мозаїка L|R + підпис
        h, w = cropL.shape[:2]
        mosaic = np.zeros((h, w*2, 3), dtype=np.uint8)
        mosaic[:, :w] = cropL
        mosaic[:,  w:] = cropR
        cv2.putText(mosaic, f"Z={Z:.1f}m  d={disp:.1f}px  cc={cc:.2f}  {w}x{h}",
                    (5, max(14, h-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # 4) шляхи збереження + JSON
        ddir = self.out_dir / time.strftime("%Y%m%d")
        ddir.mkdir(parents=True, exist_ok=True)
        base = time.strftime("%H%M%S") + f"_{int((now-int(now))*1000):03d}"
        png  = ddir / f"{base}_Z{Z:.1f}m_d{disp:.1f}px.png"
        cv2.imwrite(str(png), mosaic)

        meta = {
            "timestamp": now,
            "left_box":  [int(v) for v in (xL,yL,wL,hL)],
            "right_box": [int(v) for v in (xR,yR,wR,hR)],
            "disp": float(disp),
            "Z": float(Z),
            "crop_shape": [int(h), int(w)]
        }
        with open(ddir / f"{base}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self._recent.append((now, cx, cy))
        if self.debug: print("[snap] saved", png)
        return True

def _ncc_patch(gL, gR, cxL, cy, cxR, half):
    """NCC між квадратними латками навколо (cxL,cy) та (cxR,cy)."""
    y1L, y2L = max(0, cy-half), min(gL.shape[0], cy+half)
    x1L, x2L = max(0, cxL-half), min(gL.shape[1], cxL+half)
    y1R, y2R = max(0, cy-half), min(gR.shape[0], cy+half)
    x1R, x2R = max(0, cxR-half), min(gR.shape[1], cxR+half)
    a = gL[y1L:y2L, x1L:x2L]; b = gR[y1R:y2R, x1R:x2R]
    if a.shape != b.shape or a.size < 36:
        return -1.0
    res = cv2.matchTemplate(b.astype(np.uint8), a.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    return float(res[0,0])


def epipolar_ncc_match(gL, gR, cL, dmin, dmax, patch, ncc_min, search_pad=0):
    """
    Підбирає правий центр по епіполярі для заданого лівого центру cL=(x,y).
    Повертає (xR_best, ncc_best) або (None, -1).
    """
    xL, y = cL
    # шукаємо xR у діапазоні (для стандартної конфігурації: xR = xL - disp)
    Dmin = max(1, int(dmin) - int(search_pad))
    Dmax = int(dmax) + int(search_pad)
    half = max(3, patch//2)
    best, xr = -1.0, None
    for d in range(Dmin, Dmax+1):
        xR = xL - d
        if xR - half < 0 or xR + half >= gR.shape[1]:  # вихід за межі
            continue
        n = _ncc_patch(gL, gR, xL, y, xR, half)
        if n > best:
            best, xr = n, xR
    if best >= ncc_min:
        return xr, best
    return None, -1.0

def _ncc(a, b):
    # коеф. кореляції (NCC) через matchTemplate; обидві латки в градаціях сірого
    a = a.astype(np.uint8); b = b.astype(np.uint8)
    if a.shape != b.shape or a.shape[0] < 6 or a.shape[1] < 6:
        return 0.0
    res = cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)
    return float(res[0, 0])

# заміни open_rtsp на фабрику
def open_rtsp_any(url, width, height, reader="ffmpeg_mjpeg", ff="ffmpeg", mjpeg_q=None, ff_threads=None):

    if reader == "ffmpeg_mjpeg":
        q = mjpeg_q if mjpeg_q is not None else 6
        th = ff_threads if ff_threads is not None else 3
        return FFmpegRTSP_MJPEG(url, width, height, use_tcp=True, ffmpeg_cmd=ff, q=q, threads=th)
    if reader == "ffmpeg_raw":
        # сирий BGR через pipe (високе навантаження) — залишено опційно
        class FFmpegRTSP:
            def __init__(self, url, width, height, use_tcp=True, ffmpeg_cmd="ffmpeg"):
                self.ffmpeg_cmd = ffmpeg_cmd
                self.url = url; self.w=int(width); self.h=int(height)
                self.frame_bytes = self.w*self.h*3
                self.proc=None; self._start()
            def _start(self):
                cmd = [
                    self.ffmpeg_cmd,
                    "-rtsp_transport", "tcp" if self.use_tcp else "udp",
                    "-rtsp_flags", "prefer_tcp",
                    
                    "-analyzeduration", "1000000",
                    "-probesize", "1000000",

                    "-hide_banner", "-loglevel", "warning",     # було: quiet — нічого не видно
                    "-fflags", "nobuffer+discardcorrupt+genpts",
                    "-fflags", "nobuffer",                      # <— важливо проти фрізів
                    "-flags", "+low_delay",                     # обов’язково з плюсом
                    "-use_wallclock_as_timestamps", "1",
                    "-max_delay", "500000",


                    "-i", self.url,
                    "-an",
                    "-vf", f"scale={self.w}:{self.h}:flags=bicubic",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "-threads", str(self.threads),
                    "-q:v", str(self.q),
                    "-"
                ]
                # self.proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=None,bufsize=10**7)
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=None, bufsize=10**7
                )
            def isOpened(self): return self.proc is not None and self.proc.poll() is None
            def release(self):
                try:
                    if self.proc: self.proc.kill(); self.proc.stdout.close()
                except Exception: pass
                self.proc=None
            def read(self):
                buf=self.proc.stdout.read(self.w*self.h*3)
                if not buf or len(buf)!=self.w*self.h*3: return False,None
                frm=np.frombuffer(buf,dtype=np.uint8).reshape(self.h,self.w,3)
                return True,frm
            def grab(self): return True
            def retrieve(self): return self.read()
        return FFmpegRTSP(url, width, height, use_tcp=True, ffmpeg_cmd=ff)

    # запасний варіант — OpenCV FFmpeg
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--width", type=int, default=2880)
    ap.add_argument("--height", type=int, default=1620)
    ap.add_argument("--calib_dir", default="stereo_rtsp_out")
    ap.add_argument("--intrinsics", default=None)
    ap.add_argument("--baseline", type=float, default=None, help="база стерео в метрах (для Z у Q)")
    ap.add_argument("--rect_alpha", type=float, default=0.0, help="alpha для stereoRectify (0 — обрізати чорні поля)")
    ap.add_argument("--y_eps", type=int, default=8, help="епсилон по Y для підбору пар")
    ap.add_argument("--ignore_dist", action="store_true", help="ігнорувати дисторсію (D=0)")
    ap.add_argument("--keep_P_as_K", action="store_true", help="залишити P1=P2=K (не змінювати фокус)")
    ap.add_argument("--debug", action="store_true", help="друкувати діапазони карт і розміри")
    ap.add_argument("--autoroll", action="store_true",
                help="автоматично занулити спільний рол (за R1)")
    ap.add_argument("--roll_deg", type=float, default=0.0,
                help="додатково повернути на цей кут (°), +CW / -CCW у підсумку")
    ap.add_argument("--dmin", type=int, default=4,   help="мін. модуль диспаритету (px) — відсікає «нескінченні» хмари/блиски")
    ap.add_argument("--dmax", type=int, default=400, help="макс. модуль диспаритету (px) — відсікає зовсім близькі артефакти")
    ap.add_argument("--reader", choices=["opencv","ffmpeg_raw","ffmpeg_mjpeg"],
                    default="ffmpeg_mjpeg")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--mjpeg_q", type=int, default=6, help="якість MJPEG у пайпі (2..8; менше = краща якість)")
    ap.add_argument("--ff_threads", type=int, default=3, help="кількість потоків для MJPEG-енкодера ffmpeg")
    ap.add_argument("--snap_dir", type=str, default="detections")
    ap.add_argument("--snap_min_disp", type=float, default=1.5)
    ap.add_argument("--snap_min_cc", type=float, default=0.60)
    ap.add_argument("--snap_min_z", type=float, default=3.0)
    ap.add_argument("--snap_max_z", type=float, default=5000.0)
    ap.add_argument("--snap_cooldown", type=float, default=1.5)
    ap.add_argument("--snap_pad", type=int, default=4)
    # detection params
    ap.add_argument("--min_area", type=int, default=12, help="мінімальна площа контуру (px)")
    ap.add_argument("--max_area", type=int, default=2000, help="макс. площа контуру (px)")
    ap.add_argument("--thr_fast", type=int, default=10, help="поріг швидкого фону")
    ap.add_argument("--thr_slow", type=int, default=4, help="поріг повільного фону")
    ap.add_argument("--alpha_fast", type=float, default=0.40, help="коеф. оновлення швидкого фону")
    ap.add_argument("--alpha_slow", type=float, default=0.02, help="коеф. оновлення повільного фону")
    ap.add_argument("--min_flow", type=float, default=0.35, help="мінімальна медіанна швидкість у боксі")
    ap.add_argument("--kernel", type=int, default=2, help="розмір морф. ядра (пікселі)")
    ap.add_argument("--temporal", choices=["none","and3","maj3"], default="maj3", help="згладжування масок у часі")
    ap.add_argument("--small_targets", action="store_true", help="увімкнути пресет для дрібних об'єктів")
    ap.add_argument("--stereo_ncc_min", type=float, default=0.50, help="мін. NCC для епіполярного підбору пари")
    ap.add_argument("--stereo_patch", type=int, default=13, help="розмір квадратної латки для NCC (пікселі)")
    ap.add_argument("--stereo_search_pad", type=int, default=40, help="додатковий діапазон пошуку диспаритету (px)")
    ap.add_argument("--min_flow_small", type=float, default=0.01, help="мін. швидкість для дуже малих боксів")
    ap.add_argument("--snap_debug", action="store_true",
                help="друкувати причини, чому знімок пропущено")
    ap.add_argument("--no_rect", action="store_true", help="не выполнять ректификацию; показать сырые кадры")

    args = ap.parse_args()

    # пресет для дрібних цілей
    if args.small_targets:
        args.min_area = min(args.min_area, 10)
        args.thr_fast = min(args.thr_fast, 9)
        args.thr_slow = min(args.thr_slow, 4)
        args.min_flow = min(args.min_flow, 0.35)
        args.kernel   = min(args.kernel, 2)
        args.temporal = "maj3"

    set_ffmpeg_tcp()
    capL = open_rtsp_any(args.left, args.width, args.height, reader=args.reader, ff=args.ffmpeg, mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads)
    capR = open_rtsp_any(args.right, args.width, args.height, reader=args.reader, ff=args.ffmpeg, mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Не відкрився один із RTSP потоків")

    # Дочекаємось першого кадру з обох камер (до 8 с)
    t0 = time.time()
    while time.time() - t0 < 8:
        okL, _ = capL.read()
        okR, _ = capR.read()
        if okL and okR:
            break
        time.sleep(0.2)

    if not (okL and okR):
        print("[!] Камери не дали перших кадрів за 8с — пробую перепідключитись")
        capL.release(); capR.release()
        time.sleep(0.5)
        capL = open_rtsp_any(args.left, args.width, args.height, reader=args.reader, ff=args.ffmpeg,
                            mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads)
        capR = open_rtsp_any(args.right, args.width, args.height, reader=args.reader, ff=args.ffmpeg,
                            mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads)

    calib = load_calibration(
        args.calib_dir,
        intrinsics_npz=args.intrinsics,
        baseline_m=args.baseline,
        rect_alpha=args.rect_alpha,
        ignore_dist=args.ignore_dist,
        keep_P_as_K=args.keep_P_as_K,
        frame_size=(args.width, args.height),
        autoroll=args.autoroll,
        roll_deg=args.roll_deg,
    )

    mode = calib["mode"]
    if mode == "metric":
        map1x,map1y,map2x,map2y = calib["maps"]
        Q = calib["Q"]
    else:
        H1,H2 = calib["H"]
        Q = None

    saver = SnapshotSaver(
        args.snap_dir, args.snap_min_disp, args.snap_min_cc,
        args.snap_min_z, args.snap_max_z, args.snap_cooldown, args.snap_pad,
        debug=args.snap_debug
    )
    print("[snap_dir] ->", Path(args.snap_dir).resolve())

    bsL, bsR, Kker = make_motion_detector(kernel=args.kernel)

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
                capL = open_rtsp_any(args.left, args.width, args.height, reader=args.reader, ff=args.ffmpeg, mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads)
                capR = open_rtsp_any(args.right, args.width, args.height, reader=args.reader, ff=args.ffmpeg, mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads)
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
        if args.no_rect:
            rL, rR = fL, fR
        else:
            if mode == "metric":
                rL = cv2.remap(fL, map1x, map1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                rR = cv2.remap(fR, map2x, map2y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                rL = cv2.warpPerspective(fL, H1, (args.width, args.height), borderMode=cv2.BORDER_REPLICATE)
                rR = cv2.warpPerspective(fR, H2, (args.width, args.height), borderMode=cv2.BORDER_REPLICATE)

        # проста детекція (можна прибрати/замінити своїм кодом)
        gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
        boxesL, mL = motion_bboxes(rL, bsL, Kker, min_area=args.min_area, max_area=args.max_area, thr_fast=args.thr_fast, thr_slow=args.thr_slow, alpha_fast=args.alpha_fast, alpha_slow=args.alpha_slow, min_flow=args.min_flow, temporal=args.temporal)
        boxesR, mR = motion_bboxes(rR, bsR, Kker, min_area=args.min_area, max_area=args.max_area, thr_fast=args.thr_fast, thr_slow=args.thr_slow, alpha_fast=args.alpha_fast, alpha_slow=args.alpha_slow, min_flow=args.min_flow, temporal=args.temporal)

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
            cv2.rectangle(outL,(xL,yL),(xL+wL,yL+hL),(0,0,255),2)
            cv2.rectangle(outR,(xR,yR),(xR+wR,yR+hR),(0,0,255),2)
            cv2.circle(outL, cL, 3, (0,0,255), -1)
            cv2.circle(outR, cR, 3, (0,0,255), -1)
            label = f"d={disp:.1f}px"
            if Q is not None:
                X, Y, Z = reproject_point_Q(float(cL[0]), float(cL[1]), float(cR[0]), Q)
                label += f"  Z={Z:.1f}m"
                saver.maybe_save(rL, rR, (xL,yL,wL,hL), (xR,yR,wR,hR), disp, float(Z), now=time.time())
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
