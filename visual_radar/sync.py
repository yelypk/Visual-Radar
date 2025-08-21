
from collections import deque

def best_time_aligned(left_buf: deque, right_buf: deque, max_dt: float=0.050):
    if not left_buf or not right_buf:
        return None, None
    tL, fL = left_buf[-1]
    tR, fR = min(right_buf, key=lambda tr: abs(tr[0]-tL))
    return (fL, fR)
