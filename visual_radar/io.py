from __future__ import annotations

import logging, os, time
from typing import Optional, Tuple, Union

import numpy as np
import cv2 as cv


# ---------- small utils ----------

def now_s() -> float:
    try:
        return time.monotonic()
    except Exception:
        return time.time()


def _is_rtsp(src: Union[str, int]) -> bool:
    try:
        return isinstance(src, str) and src.strip().lower().startswith("rtsp://")
    except Exception:
        return False


def _is_device(src: Union[str, int]) -> bool:
    if isinstance(src, int):
        return True
    try:
        s = str(src).strip()
        return s.isdigit()
    except Exception:
        return False


def _pump_gui():
    try:
        cv.waitKey(1)
    except Exception:
        pass


# ---------- sane defaults for OpenCV-FFmpeg ----------

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|allowed_media_types;video|stimeout;15000000|rw_timeout;15000000|max_delay;1000000|buffer_size;1048576",
)


# ---------- FFmpegRTSP_MJPEG (совместимость, под капотом OpenCV) ----------

class FFmpegRTSP_MJPEG:
    def __init__(
        self,
        url: Union[str, int],
        width: int,
        height: int,
        ffmpeg_cmd: str = "ffmpeg",
        q: int = 6,
        threads: int = 3,
        read_timeout: float = 0.3,
    ) -> None:
        self.url = url
        self.w = int(width)
        self.h = int(height)
        self.read_timeout = float(read_timeout)
        self.log = logging.getLogger(f"visual_radar.io.ffmpeg_mjpeg.{id(self):x}")
        self.cap: Optional[cv.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        backend = cv.CAP_FFMPEG
        src: Union[str, int]
        if _is_device(self.url):
            backend = cv.CAP_ANY
            src = int(self.url) if not isinstance(self.url, int) else self.url
        else:
            src = str(self.url)

        self.cap = cv.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.url}")

        try:
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
        except Exception:
            pass

        if not _is_rtsp(self.url) and self.w and self.h:
            try:
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.w)
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.h)
            except Exception:
                pass

        self.log.info("opened via %s src=%r", "CAP_FFMPEG" if backend == cv.CAP_FFMPEG else "CAP_ANY", src)

    def isOpened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        deadline = now_s() + self.read_timeout
        while True:
            if self.cap is None or not self.cap.isOpened():
                self.reopen()
                time.sleep(0.01)

            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if ok and frame is not None:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                return True, frame, now_s()

            if now_s() > deadline:
                self.log.debug("read timeout %.0f ms url=%r", self.read_timeout * 1000, self.url)
                return False, None, None

            _pump_gui()

    def reopen(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self._open()

    def release(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def last_stderr(self) -> str:
        return ""


# ---------- универсальный ридер (opencv) ----------

class RTSPReader:
    def __init__(
        self,
        url: Union[str, int],
        width: int,
        height: int,
        cap_buffersize: Optional[int] = None,
        read_timeout: float = 0.3,
    ) -> None:
        self.url = url
        self.w = int(width)
        self.h = int(height)
        self.cap_buffersize = cap_buffersize
        self.read_timeout = float(read_timeout)
        self.log = logging.getLogger(f"visual_radar.io.opencv.{id(self):x}")
        self.cap: Optional[cv.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        is_rtsp = _is_rtsp(self.url)
        backend = cv.CAP_FFMPEG if is_rtsp else cv.CAP_ANY
        src: Union[str, int] = int(self.url) if _is_device(self.url) else str(self.url)

        self.cap = cv.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.url}")

        if self.cap_buffersize is not None:
            try:
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, int(self.cap_buffersize))
            except Exception:
                pass

        try:
            self.cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
        except Exception:
            pass

        if not is_rtsp and self.w and self.h:
            try:
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.w)
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.h)
            except Exception:
                pass

        self.log.info("opened src=%r backend=%s", src, "CAP_FFMPEG" if backend == cv.CAP_FFMPEG else "CAP_ANY")

    def isOpened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        deadline = now_s() + self.read_timeout
        while True:
            if self.cap is None or not self.cap.isOpened():
                self.log.warning("read(): backend not opened, reopen()")
                self.reopen()
                time.sleep(0.01)

            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if ok and frame is not None:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                return True, frame, now_s()

            if now_s() > deadline:
                self.log.debug("read timeout %.0f ms url=%r", self.read_timeout * 1000, self.url)
                return False, None, None

            _pump_gui()

    def reopen(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self._open()

    def release(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


# ---------- writer ----------

def make_writer(
    path: str,
    size: Tuple[int, int],
    fps: float = 15.0,
    fourcc: str = "mp4v",
    is_color: bool = True,
) -> Optional[cv.VideoWriter]:
    if not path:
        return None
    h, w = int(size[1]), int(size[0])
    try:
        four = cv.VideoWriter_fourcc(*fourcc)
    except Exception:
        four = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(path, four, float(fps), (w, h), isColor=is_color)
    if not writer.isOpened():
        four = cv.VideoWriter_fourcc(*"XVID")
        writer = cv.VideoWriter(path, four, float(fps), (w, h), isColor=is_color)
    return writer if writer.isOpened() else None


# ---------- factory ----------

def open_stream(
    url: Union[str, int],
    width: int,
    height: int,
    reader: str = "opencv",
    ffmpeg: str = "ffmpeg",
    mjpeg_q: int = 6,
    ff_threads: int = 3,
    cap_buffersize: Optional[int] = None,
    read_timeout: float = 0.3,
):
    if reader == "ffmpeg_mjpeg":
        return FFmpegRTSP_MJPEG(
            url, width, height,
            ffmpeg_cmd=ffmpeg, q=mjpeg_q, threads=ff_threads,
            read_timeout=read_timeout,
        )
    else:
        return RTSPReader(url, width, height, cap_buffersize=cap_buffersize, read_timeout=read_timeout)
