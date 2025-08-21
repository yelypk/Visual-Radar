from typing import List, Tuple
import numpy as np
import cv2 as cv
from .utils import BBox

def draw_boxes(img, boxes: List[BBox], color=(0,255,0), label=""):
    for b in boxes:
        x,y,w,h = map(int, (b.x,b.y,b.w,b.h))
        cv.rectangle(img, (x,y), (x+w,y+h), color, 2)
        if label:
            cv.putText(img, f"{label}", (x, max(0,y-5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

def stack_lr(left, right):
    return np.hstack([left, right])
