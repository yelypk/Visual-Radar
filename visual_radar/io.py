from __future__ import annotations

from typing import Optional, Tuple, Union
import os
import shutil
import subprocess
import threading
import queue
import time

import numpy as np
import cv2 as cv

from visual_radar.utils import now_s


def _is_rtsp(url: str) -> bool:
    """
    Check if the URL is an RTSP stream.
    """
    try:
        return url.lower().startswith("rtsp://")
    except Exception:
        return False


class FFmpegRTSP_MJPEG:
    """
    RTSP → ffmpeg (TCP) → MJPEG via pipe → cv2.imdecode
    API: read() -> (ok, frame, ts), isOpened(), reopen(), release()
    """

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        use_tcp: bool = True,
        ffmpeg_cmd: str = "ffmpeg",
        q: int = 6,
        threads: int = 3,
        read_timeout: float = 1.0,
    ):
        self.url = url
        self.w = int(width) if width else 0
        self.h = int(height) if height else 0
        self.use_tcp = bool(use_tcp)
        self.ffmpeg_cmd = ffmpeg_cmd
        self.q = int(q)
        self.threads = int(threads)
        self.read_timeout = float(read_timeout)

        self.proc: Optional[subprocess.Popen] = None
        self.buf = bytearray()
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

        self._start()

    def _start(self) -> None:
        """
        Start the ffmpeg process and reader thread.
        """
        if not (shutil.which(self.ffmpeg_cmd) or os.path.exists(self.ffmpeg_cmd)):
            raise RuntimeError(f"ffmpeg not found: {self.ffmpeg_cmd}")

        scale = []
        if self.w and self.h:
            scale = ["-vf", f"scale={self.w}:{self.h}:flags=bicubic"]

        cmd = [
            self.ffmpeg_cmd,
            "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp" if self.use_tcp else "udp",
            "-rtsp_flags", "prefer_tcp",
            "-rw_timeout", "5000000",          # 5s I/O timeout cross-platform
            "-analyzeduration", "10000000",
            "-probesize", "10000000",
            "-fflags", "+genpts+igndts",
            "-flags", "+low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-i", self.url,
            "-an",
            *scale,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-threads", str(self.threads),
            "-q:v", str(self.q),
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**7,
        )
        self.buf.clear()
        self._stop.clear()
        self._thr = threading.Thread(target=self._reader, daemon=True)
        self._thr.start()

    def _reader(self) -> None:
        """
        Read MJPEG frames from ffmpeg stdout and push complete JPEGs to the queue.
        """
        CHUNK = 65536
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"  # JPEG markers
        while not self._stop.is_set() and self.proc and self.proc.poll() is None:
            try:
                data = self.proc.stdout.read(CHUNK)  # type: ignore[arg-type]
                if not data:
                    time.sleep(0.01)
                    continue
                self.buf += data
                # Extract complete JPEGs from buffer
                while True:
                    i = self.buf.find(SOI)
                    if i == -1:
                        if len(self.buf) > 10 * CHUNK:
                            self.buf.clear()
                        break
                    j = self.buf.find(EOI, i + 2)
                    if j == -1:
                        if len(self.buf) > 20 * CHUNK:
                            del self.buf[:i]
                        break
                    jpg = bytes(self.buf[i : j + 2])
                    del self.buf[: j + 2]
                    try:
                        self._q.put_nowait(jpg)
                    except queue.Full:
                        pass
            except Exception:
                time.sleep(0.01)
                continue

    def isOpened(self) -> bool:
        """
        Check if the ffmpeg process is running.
        """
        return self.proc is not None and self.proc.poll() is None

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Read a frame from the MJPEG stream.
        """
        if self.proc is None or self.proc.poll() is not None:
            self.reopen()
            time.sleep(0.1)
        try:
            jpg = self._q.get(timeout=self.read_timeout)
        except queue.Empty:
            return False, None, None
        img = cv.imdecode(np.frombuffer(jpg, np.uint8), cv.IMREAD_COLOR)
        if img is None:
            return False, None, None
        return True, img, now_s()

    def reopen(self) -> None:
        """
        Restart the ffmpeg process and reader thread.
        """
        self.release()
        self._start()

    def release(self) -> None:
        """
        Stop the reader thread and kill the ffmpeg process.
        """
        try:
            self._stop.set()
            if self._thr and self._thr.is_alive():
                self._thr.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.proc:
                self.proc.kill()
                if self.proc.stdout:
                    self.proc.stdout.close()
        except Exception:
            pass
        self.proc = None
        self._thr = None
        with self._q.mutex:
            self._q.queue.clear()


class RTSPReader:
    """
    OpenCV VideoCapture with FFMPEG backend.
    API: read() -> (ok, frame, ts), reopen(), release()
    """

    def _set_default_ffmpeg_opts(self):
        """
        Set default FFMPEG options for RTSP streams.
        """
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
        """
        Open the video stream.
        """
        try:
            self._set_default_ffmpeg_opts()
        except Exception:
            pass

        self.cap = cv.VideoCapture(self.url, cv.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.url}")

        # Reduce internal buffer for RTSP
        if _is_rtsp(self.url):
            try:
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, self.cap_buffersize)
            except Exception:
                pass
            # Set new timeout properties if available
            for prop, val in [(getattr(cv, "CAP_PROP_OPEN_TIMEOUT_MSEC", None), 4000),
                              (getattr(cv, "CAP_PROP_READ_TIMEOUT_MSEC", None), 4000)]:
                if prop is not None:
                    try:
                        self.cap.set(prop, val)  # type: ignore[arg-type]
                    except Exception:
                        pass

        # Ensure BGR uint8
        try:
            self.cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
        except Exception:
            pass

        # Set frame size for local files/USB cameras
        if self.w and self.h and not _is_rtsp(self.url):
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.h)

    def isOpened(self) -> bool:
        """
        Check if the stream is open.
        """
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Read a frame from the stream.
        """
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return True, frame, now_s()

    def reopen(self) -> None:
        """
        Reopen the video stream.
        """
        try:
            self.cap.release()
        except Exception:
            pass
        self._open()

    def release(self) -> None:
        """
        Release the video stream.
        """
        try:
            self.cap.release()
        except Exception:
            pass


def open_stream(
    url: str,
    width: int,
    height: int,
    reader: str = "opencv",
    ffmpeg: str = "ffmpeg",
    mjpeg_q: int = 6,
    ff_threads: int = 3,
    cap_buffersize: Optional[int] = None,
):
    """
    Open a video stream using the specified backend.
    """
    if reader == "ffmpeg_mjpeg":
        return FFmpegRTSP_MJPEG(
            url, width, height,
            ffmpeg_cmd=ffmpeg, q=mjpeg_q, threads=ff_threads
        )
    return RTSPReader(url, width, height, cap_buffersize=cap_buffersize)


def make_writer(path: str, frame_size: Tuple[int, int], fps: float = 20.0):
    """
    Create a VideoWriter based on file extension ('.avi' → MJPG, otherwise MP4V).
    """
    w, h = map(int, frame_size)
    ext = os.path.splitext(path)[1].lower()
    fourcc = cv.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
    writer = cv.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer