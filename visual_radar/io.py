from typing import Optional, Tuple
import os, shutil, subprocess, threading, queue, time
import numpy as np
import cv2 as cv
from .utils import now_s

class FFmpegRTSP_MJPEG:
    """
    Read RTSP and transcode to MJPEG via ffmpeg -> image2pipe.
    Provides API similar to OpenCV VideoCapture.
    """
    def __init__(self, url: str, width: int, height: int, use_tcp: bool=True,
                 ffmpeg_cmd: str="ffmpeg", q: int=6, threads: int=3, read_timeout: float=0.8):
        self.ffmpeg_cmd = ffmpeg_cmd
        self.w, self.h = int(width) if width else 0, int(height) if height else 0
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
            raise RuntimeError(f"ffmpeg not found: {self.ffmpeg_cmd}")
        scale_filter = []
        if self.w and self.h:
            scale_filter = ["-vf", f"scale={self.w}:{self.h}:flags=bicubic"]
        cmd = [
            self.ffmpeg_cmd,
            "-rtsp_transport", "tcp" if self.use_tcp else "udp",
            "-rtsp_flags", "prefer_tcp",
            "-rw_timeout", "5000000",          # 5s I/O timeout
            "-stimeout",   "5000000",          # 5s socket timeout
            "-analyzeduration", "10000000",
            "-probesize", "10000000",
            "-hide_banner", "-loglevel", "warning",
            "-fflags", "+genpts+igndts",       # мягче, чем nobuffer+discardcorrupt
            "-flags", "+low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-i", self.url,
            "-an",
            *scale_filter,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-threads", str(self.threads),
            "-q:v", str(self.q),
            "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, bufsize=10**7)
        self.buf.clear()
        self._stop.clear()
        self._thr = threading.Thread(target=self._reader, daemon=True); self._thr.start()

    def _reader(self):
        CHUNK = 65536; SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        while not self._stop.is_set() and self.proc and self.proc.poll() is None:
            try:
                data = self.proc.stdout.read(CHUNK)
                if not data:
                    time.sleep(0.01); continue
                self.buf += data
                while True:
                    i = self.buf.find(SOI)
                    if i == -1:
                        if len(self.buf) > 10*CHUNK: self.buf.clear()
                        break
                    j = self.buf.find(EOI, i+2)
                    if j == -1:
                        if len(self.buf) > 20*CHUNK: del self.buf[:i]
                        break
                    jpg = bytes(self.buf[i:j+2]); del self.buf[:j+2]
                    try:
                        self._q.put_nowait(jpg)
                    except queue.Full:
                        pass
            except Exception:
                time.sleep(0.01); continue

    def isOpened(self): 
        return self.proc is not None and self.proc.poll() is None

    def release(self):
        try:
            self._stop.set()
            if self._thr and self._thr.is_alive(): self._thr.join(timeout=1.0)
        except Exception: 
            pass
        try:
            if self.proc:
                self.proc.kill()
                if self.proc.stdout: self.proc.stdout.close()
        except Exception: 
            pass
        self.proc=None; self._thr=None
        with self._q.mutex:
            self._q.queue.clear()

    def read(self):
        if self.proc is None or self.proc.poll() is not None:
            self.release(); self._start(); time.sleep(0.1)
        try:
            jpg = self._q.get(timeout=self.read_timeout)
        except queue.Empty:
            return False, None, None
        img = cv.imdecode(np.frombuffer(jpg, np.uint8), cv.IMREAD_COLOR)
        if img is None:
            return False, None, None
        ts = now_s()
        return True, img, ts

class RTSPReader:
    def __init__(self, url: str, width: Optional[int], height: Optional[int]):
        self.cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
        if width:  self.cap.set(cv.CAP_PROP_FRAME_WIDTH,  width)
        if height: self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.ok = self.cap.isOpened()
        if not self.ok:
            raise RuntimeError(f"Cannot open stream: {url}")
    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, None
        ts = now_s()
        return True, frame, ts
    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

def open_stream(url: str, width: int, height: int,
                reader: str="opencv", ffmpeg: str="ffmpeg",
                mjpeg_q: int=6, ff_threads: int=3):
    if reader == "ffmpeg_mjpeg":
        return FFmpegRTSP_MJPEG(url, width, height, ffmpeg_cmd=ffmpeg, q=mjpeg_q, threads=ff_threads)
    return RTSPReader(url, width, height)

def make_writer(path: str, frame_size: Tuple[int,int], fps: float=20.0):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    w,h = frame_size
    return cv.VideoWriter(path, fourcc, fps, (w,h))