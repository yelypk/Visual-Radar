
from collections import namedtuple
import time

BBox = namedtuple("BBox", "x y w h area cx cy")

def now_s():
    return time.time()

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def to_bbox(x, y, w, h):
    cx = x + w/2.0; cy = y + h/2.0
    return BBox(x,y,w,h,w*h,cx,cy)
