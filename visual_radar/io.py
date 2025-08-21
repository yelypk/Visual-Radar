from typing import Optional, Tuple
import cv2 as cv
from .utils import now_s

class RTSPReader:
    def __init__(self, url: str, width: Optional[int], height: Optional[int]):
        self.cap = cv.VideoCapture(url)
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

def make_writer(path: str, frame_size: Tuple[int,int], fps: float=20.0):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    w,h = frame_size
    return cv.VideoWriter(path, fourcc, fps, (w,h))
