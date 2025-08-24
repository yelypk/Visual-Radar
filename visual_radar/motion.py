from __future__ import annotations

from typing import List, Tuple
from pathlib import Path
import time
import numpy as np
import cv2 as cv

from visual_radar.utils import BBox
from visual_radar.io import open_stream  # RTSP reader (opencv | ffmpeg_mjpeg)

# -----------------------------
# Helpers
# -----------------------------
def as_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to GRAY uint8."""
    if img is None:
        raise ValueError("as_gray: None image")
    if img.ndim == 2:
        g = img
    else:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return g


def _kernel_from_area(min_area: int, boost: float = 1.0) -> Tuple[int, int]:
    """
    Choose a morphology kernel size from area:
    min_area ~ k^2  ->  k ~ sqrt(min_area).
    """
    k = int(max(3, np.sqrt(max(1, float(min_area))) * 0.50 * float(boost)))
    if k % 2 == 0:
        k += 1
    k = int(np.clip(k, 3, 21))
    return k, k


def _morph_clean(mask: np.ndarray, min_area: int, size_aware_morph: bool = True) -> np.ndarray:
    """Open+Close with size-aware kernel to denoise the mask."""
    if mask is None or mask.size == 0:
        return mask
    mask = (mask > 0).astype(np.uint8) * 255
    if size_aware_morph:
        kx, ky = _kernel_from_area(min_area)
    else:
        kx = ky = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kx, ky))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _boxes_from_mask(mask: np.ndarray, min_area: int, max_area: int) -> List[BBox]:
    """Extract bounding boxes from a binary mask."""
    boxes: List[BBox] = []
    if mask is None or mask.size == 0:
        return boxes
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        area = int(w) * int(h)
        if area < max(1, int(min_area)):
            continue
        if max_area and area > int(max_area):
            continue
        boxes.append(BBox(float(x), float(y), float(w), float(h)))
    return boxes


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union for two boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / union


# -----------------------------
# Background models
# -----------------------------
class DualBGModel:
    """
    Two background models:
      - fast: short memory → detects fast motion (birds, cars)
      - slow: long memory → keeps slow drift (boats, water)
    Both implemented with MOG2.
    """
    def __init__(self, shape_hw: Tuple[int, int]):
        h, w = int(shape_hw[0]), int(shape_hw[1])
        self.fast = cv.createBackgroundSubtractorMOG2(history=50,  varThreshold=16, detectShadows=True)
        self.slow = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=12, detectShadows=True)

    def apply_fast(self, gray: np.ndarray, lr: float = -1.0) -> np.ndarray:
        fg = self.fast.apply(gray, learningRate=lr)
        return (fg > 200).astype(np.uint8) * 255  # drop shadows (~127)

    def apply_slow(self, gray: np.ndarray, lr: float = -1.0) -> np.ndarray:
        fg = self.slow.apply(gray, learningRate=lr)
        return (fg > 200).astype(np.uint8) * 255


def find_motion_bboxes(
    gray: np.ndarray,
    bg: DualBGModel,
    min_area: int,
    max_area: int,
    thr_fast: float,
    thr_slow: float,
    use_clahe: bool = True,
    size_aware_morph: bool = True,
    learning_rate: float = -1.0,   # NEW: pass-through LR to keep slow targets in FG
) -> Tuple[np.ndarray, List[BBox]]:
    """
    Weighted union of fast/slow MOG2 masks → morph → boxes.
    If you want slow movers (boats) to remain foreground, pass a small learning_rate, e.g. 0.0005.
    """
    g = gray
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    # Use the given learning rate for both models
    mf = bg.apply_fast(g, lr=learning_rate)
    ms = bg.apply_slow(g, lr=learning_rate)

    alpha = float(max(0.0, thr_fast))
    beta  = float(max(0.0, thr_slow))
    denom = max(1e-6, alpha + beta)
    alpha /= denom; beta /= denom

    mix = cv.addWeighted(mf, alpha, ms, beta, 0.0)
    mix = (mix > 0).astype(np.uint8) * 255

    mask = _morph_clean(mix, int(min_area), bool(size_aware_morph))
    boxes = _boxes_from_mask(mask, int(min_area), int(max_area))
    return mask, boxes


def make_masks_static_and_slow(
    gray: np.ndarray,
    bg: DualBGModel,
    thr_fast: float,
    thr_slow: float,
    use_clahe: bool = True,
    kernel_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Utility that returns (static_mask, slow_mask) for debugging."""
    g = gray
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    mf = bg.apply_fast(g)
    ms = bg.apply_slow(g)

    k = max(3, int(kernel_size) | 1)
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

    fast_bin = (mf > 0).astype(np.uint8) * 255
    slow_bin = (ms > 0).astype(np.uint8) * 255

    static = cv.bitwise_not(fast_bin)
    static = cv.morphologyEx(static, cv.MORPH_OPEN, ker, iterations=1)

    slow = cv.morphologyEx(slow_bin, cv.MORPH_OPEN, ker, iterations=1)
    return static, slow


def _draw_boxes(frame: np.ndarray, boxes_speeds: List[Tuple[Tuple[int, int, int, int], float, float]]) -> None:
    """Draw boxes with area and speed."""
    for (x1, y1, x2, y2), area, speed in boxes_speeds:
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(
            frame,
            f"{int(area)}px  {int(speed)}px/s",
            (x1, max(15, y1 - 5)),
            cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv.LINE_AA
        )


# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Motion detection for fast (birds) and slow (boats) targets")
    # NEW: support legacy flags --left/--right and alias --timeout
    ap.add_argument("--url", help="RTSP URL (single stream)")
    ap.add_argument("--left", help="Alias for --url (use LEFT stream if both given)")
    ap.add_argument("--right", help="Alias for --url (used if --url/--left missing)")

    ap.add_argument("--reader", default="ffmpeg_mjpeg", choices=["ffmpeg_mjpeg", "opencv"])
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)

    # Reader tuning
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--mjpeg-q", type=int, default=6)
    ap.add_argument("--ff-threads", type=int, default=3)
    ap.add_argument("--cap-buffersize", type=int, default=1)
    ap.add_argument("--read-timeout", type=float, default=0.2, dest="read_timeout")
    ap.add_argument("--timeout", type=float, dest="read_timeout", help="Alias for --read-timeout")

    # BG models
    ap.add_argument("--history-fast", type=int, default=50)
    ap.add_argument("--history-slow", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=-1.0,
                    help="-1 = auto; small value like 0.0005 keeps slow targets in foreground")

    # Threshold mixing
    ap.add_argument("--thr-fast", type=float, default=0.65)
    ap.add_argument("--thr-slow", type=float, default=0.35)
    ap.add_argument("--clahe", action="store_true")

    # Morph/filters
    ap.add_argument("--min-area", type=int, default=120)
    ap.add_argument("--max-area-frac", type=float, default=0.02)
    ap.add_argument("--fill-thresh", type=float, default=0.35)
    ap.add_argument("--size-aware-morph", action="store_true")

    # ROI (ignore sky)
    ap.add_argument("--roi-top-frac", type=float, default=0.35)

    # Slow branch (optical flow)
    ap.add_argument("--slow-flow", action="store_true")
    ap.add_argument("--flow-thresh", type=float, default=0.25)
    ap.add_argument("--flow-ema", type=float, default=0.9)
    ap.add_argument("--flow-k", type=int, default=5)

    # Speed filter
    ap.add_argument("--min-speed", type=float, default=40.0)

    # Output
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-snaps", action="store_true")
    ap.add_argument("--save-dir", default="detections")

    args = ap.parse_args()

    # Resolve URL from --url / --left / --right
    if not args.url:
        args.url = args.left or args.right
    if not args.url:
        ap.error("Provide --url (or legacy flags --left / --right)")

    frame_size = (args.width, args.height)

    # Configure BG histories
    bg = DualBGModel((args.height, args.width))
    bg.fast.setHistory(args.history_fast)
    bg.slow.setHistory(args.history_slow)

    # Open stream (ffmpeg_mjpeg is recommended for RTSP)
    r = open_stream(
        url=args.url, width=args.width, height=args.height,
        reader=args.reader, ffmpeg=args.ffmpeg, mjpeg_q=args.mjpeg_q,
        ff_threads=args.ff_threads, cap_buffersize=args.cap_buffersize,
        read_timeout=args.read_timeout,
    )

    stall = 0
    last_boxes: List[Tuple[int, int, int, int]] = []
    last_centroids: List[Tuple[float, float]] = []
    last_ts: float | None = None

    flow_prev = None
    flow_accum = None
    k_flow = cv.getStructuringElement(cv.MORPH_ELLIPSE, (max(3, args.flow_k | 1), max(3, args.flow_k | 1)))

    outdir = Path(args.save_dir)
    if args.save_snaps:
        outdir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame, ts = r.read()
            if not ok or frame is None:
                # Non-blocking read: after ~2s of silence, try to reopen ffmpeg
                stall += 1
                if stall >= int(max(1.0, 2.0 / max(args.read_timeout, 1e-3))):
                    try:
                        print("[reader] reopen…")
                        r.reopen()
                    except Exception:
                        pass
                    stall = 0
                continue
            stall = 0

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv.resize(frame, frame_size)

            H, W = frame.shape[:2]
            gray = as_gray(frame)

            # --- base motion mask: weighted fast+slow MOG2
            mask, _ = find_motion_bboxes(
                gray, bg,
                min_area=args.min_area,
                max_area=int(args.max_area_frac * (W * H)),
                thr_fast=args.thr_fast, thr_slow=args.thr_slow,
                use_clahe=args.clahe,
                size_aware_morph=args.size_aware_morph,
                learning_rate=args.learning_rate,  # <- use LR from CLI
            )

            # --- ROI: ignore sky (top portion of the frame)
            roi = np.zeros((H, W), np.uint8)
            horizon = int(args.roi_top_frac * H)
            cv.rectangle(roi, (0, horizon), (W, H), 255, -1)
            mask = cv.bitwise_and(mask, roi)

            # --- slow branch: optical flow (Farnebäck) to catch slow movers (boats)
            slow_mask = np.zeros_like(mask)
            if args.slow_flow:
                if flow_prev is None:
                    flow_prev = gray.copy()
                else:
                    flow = cv.calcOpticalFlowFarneback(flow_prev, gray, None,
                                                       0.5, 3, 15, 3, 5, 1.2, 0)
                    mag = cv.magnitude(flow[..., 0], flow[..., 1])  # px/frame
                    slow_mask = (mag > args.flow_thresh).astype(np.uint8) * 255
                    slow_mask = cv.bitwise_and(slow_mask, roi)
                    slow_mask = cv.morphologyEx(slow_mask, cv.MORPH_OPEN,  k_flow, iterations=1)
                    slow_mask = cv.morphologyEx(slow_mask, cv.MORPH_CLOSE, k_flow, iterations=1)
                    if flow_accum is None:
                        flow_accum = slow_mask.astype(np.float32)
                    else:
                        flow_accum = args.flow_ema * flow_accum + (1.0 - args.flow_ema) * slow_mask
                    slow_mask = (flow_accum > 127).astype(np.uint8) * 255
                flow_prev = gray.copy()

            # --- union of base motion and slow-flow masks
            mask = cv.bitwise_or(mask, slow_mask)

            # --- final cleanup and boxes
            mask = _morph_clean(mask, args.min_area, args.size_aware_morph)
            boxes = _boxes_from_mask(mask, args.min_area, int(args.max_area_frac * (W * H)))

            # --- simple association with previous frame to estimate speed
            dets: List[Tuple[Tuple[int, int, int, int], float, float]] = []
            if last_ts is not None and last_boxes:
                dt = max(1e-3, (ts - last_ts) if ts else 1 / 25.0)
                used = set()
                for b in boxes:
                    x1, y1, w, h = map(int, (b.x, b.y, b.w, b.h))
                    x2, y2 = x1 + w, y1 + h
                    best_iou, best_j = 0.0, -1
                    for j, prev in enumerate(last_boxes):
                        if j in used:
                            continue
                        v = _iou((x1, y1, x2, y2), prev)
                        if v > best_iou:
                            best_iou, best_j = v, j
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    speed = 0.0
                    if best_iou > 0.1:
                        px1, py1, px2, py2 = last_boxes[best_j]
                        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                        used.add(best_j)
                        speed = float(np.hypot(cx - pcx, cy - pcy) / dt)
                    elif last_centroids:
                        dists = [np.hypot(cx - lx, cy - ly) for (lx, ly) in last_centroids]
                        jmin = int(np.argmin(dists))
                        speed = float(dists[jmin] / dt)
                    dets.append(((x1, y1, x2, y2), float(w * h), speed))
            else:
                for b in boxes:
                    x1, y1, w, h = map(int, (b.x, b.y, b.w, b.h))
                    dets.append(((x1, y1, x1 + w, y1 + h), float(w * h), 0.0))

            # --- final selection: fast OR (slow if overlapping slow_mask)
            selected: List[Tuple[Tuple[int, int, int, int], float, float]] = []
            for (x1, y1, x2, y2), area, speed in dets:
                pass_speed = speed >= args.min_speed
                pass_slow = False
                if args.slow_flow:
                    sx1, sy1 = max(0, x1), max(0, y1)
                    sx2, sy2 = min(W - 1, x2), min(H - 1, y2)
                    if sx2 > sx1 and sy2 > sy1:
                        sm = slow_mask[sy1:sy2, sx1:sx2]
                        pass_slow = (cv.countNonZero(sm) / float(sm.size + 1e-6)) > 0.05
                if pass_speed or pass_slow:
                    selected.append(((x1, y1, x2, y2), area, speed))

            # --- draw & output
            _draw_boxes(frame, selected)
            if args.save_snaps and selected:
                ts_ms = int((ts if ts else time.time()) * 1000)
                cv.imwrite(str(outdir / f"det_{ts_ms}.jpg"), frame)

            if args.show:
                cv.imshow("frame", frame)
                cv.imshow("mask", mask)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            # --- update history
            last_boxes = [tuple(map(int, (b.x, b.y, b.x + b.w, b.y + b.h))) for b in boxes]
            last_centroids = [((bx1 + bx2) / 2.0, (by1 + by2) / 2.0) for (bx1, by1, bx2, by2) in last_boxes]
            last_ts = ts if ts else time.time()

    except KeyboardInterrupt:
        pass
    finally:
        try:
            r.release()
        except Exception:
            pass
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
