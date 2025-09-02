from __future__ import annotations

from asyncio.log import logger
import logging
from typing import Optional, Tuple
import os
import shutil
import subprocess
import threading
import queue
import time
from pathlib import Path


import numpy as np
import cv2 as cv

from visual_radar.utils import now_s

log = logging.getLogger("visual_radar.io.ffmpeg_mjpeg")

def _is_rtsp(url: str) -> bool:
    try:
        return url.lower().startswith("rtsp://")
    except Exception:
        return False

def now_s() -> float:
    return time.time()


class FFmpegRTSP_MJPEG:
    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        use_tcp: bool = True,
        ffmpeg_cmd: str = "ffmpeg",
        q: int = 6,
        threads: int = 3,
        read_timeout: float = 0.2,
        stimeout_s: float = 5.0,          
        capture_stderr: bool = True,      
        name: str = "",                   
    ):
        self.url = url
        self.w = int(width) if width else 0
        self.h = int(height) if height else 0
        self.use_tcp = bool(use_tcp)
        self.ffmpeg_cmd = ffmpeg_cmd
        self.q = int(q)
        self.threads = int(threads)
        self.read_timeout = float(read_timeout)
        self.stimeout_us = int(max(0.0, stimeout_s) * 1_000_000)
        self.capture_stderr = capture_stderr
        self.log = log.getChild(name or f"{id(self):x}")

        self.proc: Optional[subprocess.Popen] = None
        self.buf = bytearray()
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._err_thr: Optional[threading.Thread] = None
        self._stderr_tail = []
        self._stderr_lock = threading.Lock()

        self._frames_ok = 0
        self._frames_dropped_q = 0
        self._frames_bad_jpeg = 0
        self._bytes_in = 0
        self._last_stat_t = now_s()
        self._consec_timeouts = 0

        self._rtsp_timeout_key = self._choose_rtsp_timeout_key()
        self.log.info("RTSP timeout key selected: %s",
                    f"-{self._rtsp_timeout_key}" if self._rtsp_timeout_key else "<none>")

        self._start()

    def _choose_rtsp_timeout_key(self) -> Optional[str]:
        candidates = ("stimeout", "rw_timeout", "timeout")
        try:
            out = subprocess.run(
                [self.ffmpeg_cmd, "-hide_banner", "-loglevel", "quiet", "-h", "protocol=rtsp"],
                capture_output=True, text=True, timeout=2.0
            )
            text = (out.stdout or "") + (out.stderr or "")
            for key in candidates:
                if key in text:
                    return key
        except Exception:
            pass
        try:
            out = subprocess.run(
                [self.ffmpeg_cmd, "-hide_banner", "-loglevel", "quiet", "-h"],
                capture_output=True, text=True, timeout=2.0
            )
            text = (out.stdout or "") + (out.stderr or "")
            for key in candidates:
                if key in text:
                    return key
        except Exception:
            pass
        return None


    def _build_cmd(self) -> list:
        scale = []
        if self.w and self.h:
            scale = ["-vf", f"scale={self.w}:{self.h}:flags=bicubic"]

        cmd = [
            self.ffmpeg_cmd,
            "-hide_banner", "-loglevel", "warning", "-nostats", "-nostdin",
            "-rtsp_transport", "tcp" if self.use_tcp else "udp",
            "-rtsp_flags", "prefer_tcp",               
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-analyzeduration", "1000000",              
            "-probesize", "1000000",                    
        ]

        if self._rtsp_timeout_key:
            cmd += [f"-{self._rtsp_timeout_key}", str(self.stimeout_us)]

        cmd += ["-i", self.url, "-an"]
        cmd += scale
        cmd += [
            "-f", "mjpeg",
            "-c:v", "mjpeg",
            "-threads", str(self.threads),
            "-q:v", str(self.q),
            "-"
        ]
        return cmd


    def _start(self) -> None:
        if not (shutil.which(self.ffmpeg_cmd) or os.path.exists(self.ffmpeg_cmd)):
            raise RuntimeError(f"ffmpeg not found: {self.ffmpeg_cmd}")

        scale = []
        if self.w and self.h:
            scale = ["-vf", f"scale={self.w}:{self.h}:flags=bicubic"]

        cmd = [
            self.ffmpeg_cmd,
            "-hide_banner", "-loglevel", "warning", "-nostats", "-nostdin",
            "-rtsp_transport", "tcp" if self.use_tcp else "udp",]
        if self._rtsp_timeout_key:
            cmd += [f"-{self._rtsp_timeout_key}", str(self.stimeout_us)]


        cmd += ["-fflags", "nobuffer",
            "-flags", "low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-i", self.url,
            "-an",
            *scale,
            "-f", "mjpeg",
            "-c:v", "mjpeg",
            "-threads", str(self.threads),
            "-q:v", str(self.q),
            "-",
        ]
        self.log.info("spawn ffmpeg: %s", " ".join(map(str, cmd)))

        stderr_target = subprocess.PIPE if self.capture_stderr else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_target, bufsize=0
        )
        self.buf.clear()
        self._stop.clear()
        self._thr = threading.Thread(target=self._reader, daemon=True)
        self._thr.start()
        self.log.debug("reader thread started")

        if self.capture_stderr and self.proc.stderr is not None:
            self._err_thr = threading.Thread(target=self._stderr_reader, daemon=True)
            self._err_thr.start()
            self.log.debug("stderr reader thread started")


    def _stderr_reader(self) -> None:
        MAX_LINES = 200
        try:
            assert self.proc and self.proc.stderr
            for line in iter(self.proc.stderr.readline, b""):
                s = line.decode("utf-8", "replace").rstrip()
                with self._stderr_lock:
                    self._stderr_tail.append(s)
                    if len(self._stderr_tail) > MAX_LINES:
                        self._stderr_tail = self._stderr_tail[-MAX_LINES:]
                if self._stop.is_set():
                    break
        except Exception as e:
            self.log.debug("stderr reader stopped: %r", e)

    def last_stderr(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_tail[-50:])


    def _log_stats_periodically(self) -> None:
        t = now_s()
        if t - self._last_stat_t >= 5.0:
            fps = self._frames_ok / max(1e-9, t - self._last_stat_t)
            self.log.info(
                "stats: fps=%.1f ok=%d dropped_q=%d bad_jpeg=%d bytes=%d",
                fps, self._frames_ok, self._frames_dropped_q, self._frames_bad_jpeg, self._bytes_in
            )
            self._frames_ok = 0
            self._frames_dropped_q = 0
            self._frames_bad_jpeg = 0
            self._bytes_in = 0
            self._last_stat_t = t

    def _reader(self) -> None:
        CHUNK = 65536
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        stdout = self.proc.stdout if self.proc else None

        def _stats():
            t = time.time()
            if t - self._last_stat_t >= 5.0:
                fps = self._frames_ok / max(1e-9, t - self._last_stat_t)
                self.log.info("stats: fps=%.1f ok=%d dropped_q=%d bad_jpeg=%d bytes=%d",
                            fps, self._frames_ok, self._frames_dropped_q,
                            self._frames_bad_jpeg, self._bytes_in)
                self._frames_ok = self._frames_dropped_q = self._frames_bad_jpeg = 0
                self._bytes_in = 0
                self._last_stat_t = t

        while not self._stop.is_set():
            if self.proc is None or self.proc.poll() is not None:
                rc = None if self.proc is None else self.proc.poll()
                self.log.warning("ffmpeg exited early (rc=%s). stderr tail:\n%s", rc, self.last_stderr())
                break
            if stdout is None:
                self.log.error("ffmpeg stdout is None")
                break

            try:
                data = stdout.read(CHUNK)
                if not data:
                    time.sleep(0.01)
                    continue

                self._bytes_in += len(data)
                self.buf += data

                while True:
                    i = self.buf.find(SOI)
                    if i == -1:
                        if len(self.buf) > 10 * CHUNK:
                            self.log.debug("buffer purge (no SOI, size=%d)", len(self.buf))
                            self.buf.clear()
                        break
                    j = self.buf.find(EOI, i + 2)
                    if j == -1:
                        if len(self.buf) > 20 * CHUNK:
                            self.log.debug("buffer trim (no EOI, size=%d)", len(self.buf))
                            del self.buf[:i]
                        break
                    jpg = bytes(self.buf[i:j + 2])
                    del self.buf[:j + 2]
                    try:
                        self._q.put_nowait(jpg)
                    except queue.Full:
                        self._frames_dropped_q += 1
                        if self._frames_dropped_q % 20 == 1:
                            self.log.debug("queue full, dropping (dropped=%d)", self._frames_dropped_q)

                _stats()

            except Exception as e:
                self.log.warning("reader exception: %r", e)
                time.sleep(0.01)
                continue

        self.log.debug("reader thread finished")

    def last_stderr(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_tail[-50:])

    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def read(self):
        if self.proc is None or self.proc.poll() is not None:
            self.log.warning("read(): ffmpeg not running, reopen()")
            self.reopen()
            time.sleep(0.05)

        try:
            jpg = self._q.get(timeout=self.read_timeout)
            self._consec_timeouts = 0
        except queue.Empty:
            self._consec_timeouts += 1
            if self._consec_timeouts in (5, 20, 100):
                self.log.warning(
                    "no frames for %.1fs (timeouts=%d). rc=%s\nstderr tail:\n%s",
                    self._consec_timeouts * self.read_timeout,
                    self._consec_timeouts,
                    None if self.proc is None else self.proc.poll(),
                    self.last_stderr(),
                )
            return False, None, None

        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv.imdecode(arr, cv.IMREAD_COLOR)
        if frame is None:
            self._frames_bad_jpeg += 1
            if self._frames_bad_jpeg % 10 == 1:
                self.log.debug("imdecode failed (bad_jpeg=%d, size=%d)", self._frames_bad_jpeg, len(jpg))
            return False, None, None

        self._frames_ok += 1
        return True, frame, time.time()

    def reopen(self) -> None:
        self.log.info("reopen ffmpeg")
        self.release()
        self._start()

    def release(self) -> None:
        self._stop.set()
        try:
            if self.proc:
                try:
                    self.proc.terminate(); self.proc.wait(timeout=0.5)
                except Exception:
                    pass
                try:
                    self.proc.kill()
                except Exception:
                    pass
                try:
                    if self.proc.stdout: self.proc.stdout.close()
                except Exception:
                    pass
                try:
                    if self.proc.stderr: self.proc.stderr.close()
                except Exception:
                    pass
        except Exception as e:
            self.log.debug("release: proc close error: %r", e)

        try:
            if self._thr and self._thr.is_alive(): self._thr.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._err_thr and self._err_thr.is_alive(): self._err_thr.join(timeout=0.5)
        except Exception:
            pass

        self.proc = None; self._thr = None; self._err_thr = None
        with self._q.mutex: self._q.queue.clear()
        self.buf.clear()
        self.log.info("released")
    
class RTSPReader:
    """
    OpenCV VideoCapture with FFMPEG backend.
    API: read() -> (ok, frame, ts), reopen(), release()
    """
    def _set_default_ffmpeg_opts(self):
        env = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        if not env:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|buffer_size;1048576"
            )

    def __init__(self, url: str, width: Optional[int], height: Optional[int], cap_buffersize: Optional[int] = None):
        self.url = url
        self.w = int(width) if width else 0
        self.h = int(height) if height else 0
        self.cap_buffersize = int(cap_buffersize) if cap_buffersize is not None else 1
        self._open()

    def _open(self):
        try:
            self._set_default_ffmpeg_opts()
        except Exception:
            pass

        self.cap = cv.VideoCapture(self.url, cv.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.url}")

        if _is_rtsp(self.url):
            try:
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, self.cap_buffersize)
            except Exception:
                pass
            for prop, val in [(getattr(cv, "CAP_PROP_OPEN_TIMEOUT_MSEC", None), 4000),
                              (getattr(cv, "CAP_PROP_READ_TIMEOUT_MSEC", None), 4000)]:
                if prop is not None:
                    try:
                        self.cap.set(prop, val)  # type: ignore[arg-type]
                    except Exception:
                        pass

        try:
            self.cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
        except Exception:
            pass

        if self.w and self.h and not _is_rtsp(self.url):
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.h)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return True, frame, now_s()

    def reopen(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        self._open()

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


def make_writer(
    path: str,
    size: Tuple[int, int],
    fps: float = 15.0,
    fourcc: Optional[str] = None,
    is_color: bool = True,
) -> cv.VideoWriter:
    """
    Create a cv2.VideoWriter with safe defaults for MP4/AVI/MKV.
    Tries several FOURCCs until one opens successfully.
    """
    # Ensure directory exists
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    w, h = int(size[0]), int(size[1])
    ext = p.suffix.lower()

    # Choose default FOURCC by extension (OpenCV-friendly)
    if fourcc is None:
        if ext in (".mp4", ".m4v", ".mov", ".mkv"):
            fourcc = "mp4v"   # widely available
        elif ext in (".avi",):
            fourcc = "XVID"
        else:
            fourcc = "mp4v"

    # Try a small pool of codecs in order
    candidates = [fourcc, "mp4v", "avc1", "H264", "MJPG", "XVID"]
    tried = set()
    for code in candidates:
        if not code or code in tried:
            continue
        tried.add(code)
        writer = cv.VideoWriter(
            str(p),
            cv.VideoWriter_fourcc(*code),
            float(fps),
            (w, h),
            isColor=bool(is_color),
        )
        if writer.isOpened():
            return writer

    # Last-resort dummy writer (will fail later with clear message)
    writer = cv.VideoWriter(
        str(p),
        cv.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
        isColor=bool(is_color),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for '{path}'. "
                           f"Try another extension (.mp4/.avi) or install codecs.")
    return writer

def open_stream(
    url: str,
    width: int,
    height: int,
    reader: str = "opencv",
    ffmpeg: str = "ffmpeg",
    mjpeg_q: int = 6,
    ff_threads: int = 3,
    cap_buffersize: Optional[int] = None,
    read_timeout: float = 0.2,  # pass to FFmpegRTSP_MJPEG
):
    if reader == "ffmpeg_mjpeg":
        return FFmpegRTSP_MJPEG(
            url, width, height,
            ffmpeg_cmd=ffmpeg, q=mjpeg_q, threads=ff_threads,
            read_timeout=read_timeout,
        )
    return RTSPReader(url, width, height, cap_buffersize=cap_buffersize)
