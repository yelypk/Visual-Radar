from pathlib import Path
from collections import deque
import argparse
import time
from typing import Optional
import numpy as np
import cv2 as cv

from .config import AppConfig, SMDParams
from .io import open_stream, make_writer
from .sync import best_time_aligned
from .stereo import rectified_pair
from .calibration import load_calibration
from .visualize import draw_boxes, stack_lr, imshow_resized
from .snapshots import SnapshotSaver
from .detector import StereoMotionDetector
from .tracks import BoxTracker

# --------------------------- local helpers ---------------------------------

def warmup(reader, name: str, timeout_s: float = 8.0):
    """
    Читает из reader до первого успешного кадра или до timeout.
    Возвращает (ok, frame, timestamp).
    """
    t0 = time.time()
    reads = 0
    while time.time() - t0 < timeout_s:
        ok, frame, ts = reader.read()
        reads += 1
        if ok:
            try:
                h, w = frame.shape[:2]
                ch = frame.shape[2] if frame.ndim == 3 else 1
                print(f"[warmup] {name}: got first frame ({w}, {h}, {ch}) after {reads} reads")
            except Exception:
                print(f"[warmup] {name}: got first frame after {reads} reads")
            return True, frame, ts
    print(f"[warmup] {name}: no frames within {timeout_s} s")
    return False, None, None


# ----------------------------- ARGS / CONFIG ------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Visual Radar")

    # IO
    p.add_argument("--left", required=True, help="Left RTSP/URL")
    p.add_argument("--right", required=True, help="Right RTSP/URL")
    p.add_argument("--reader", default="opencv", choices=["opencv", "ffmpeg_mjpeg"])
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (for ffmpeg_mjpeg)")
    p.add_argument("--mjpeg_q", type=int, default=6, help="MJPEG quality (ffmpeg_mjpeg)")
    p.add_argument("--ff_threads", type=int, default=4, help="ffmpeg threads (ffmpeg_mjpeg)")

    # Resolution / display
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--native_resolution", action="store_true", help="Use native stream size, ignore --width/--height")
    p.add_argument("--display", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--display_max_w", type=int, default=1920)
    p.add_argument("--display_max_h", type=int, default=1080)
    p.add_argument("--save_vis", type=str, default="", help="Path to save stacked visualization video (mp4/avi)")

    # Calibration
    p.add_argument("--calib_dir", type=str, default="", help="stereo calibration dir (rectification)")
    p.add_argument("--intrinsics", type=str, default="", help="intrinsics npz (optional)")
    p.add_argument("--baseline", type=float, default=0.0, help="baseline meters override (optional)")

    # Stereo / geometry gates
    p.add_argument("--y_eps", type=float, default=4.0)
    p.add_argument("--dmin", type=float, default=-300.0)
    p.add_argument("--dmax", type=float, default=300.0)

    # Motion / thresholds
    p.add_argument("--min_area", type=int, default=25)
    p.add_argument("--max_area", type=int, default=0)
    p.add_argument("--thr_fast", type=float, default=None)
    p.add_argument("--thr_slow", type=float, default=None)
    p.add_argument("--min_flow", type=float, default=1.0)
    p.add_argument("--min_flow_small", type=float, default=0.3)

    # Stereo NCC
    p.add_argument("--stereo_patch", type=int, default=7)
    p.add_argument("--stereo_search_pad", type=int, default=80)
    p.add_argument("--stereo_ncc_min", type=float, default=0.60)

    # Night / denoise / morph / ROI-crop
    p.add_argument("--night_auto", action="store_true")
    p.add_argument("--night_luma_thr", type=float, default=50.0)
    p.add_argument("--min_area_night_mult", type=float, default=12.0)
    p.add_argument("--noise_k_fast", type=float, default=6.0)
    p.add_argument("--noise_k_slow", type=float, default=4.0)
    p.add_argument("--morph_open_day", type=int, default=3)
    p.add_argument("--morph_open_night", type=int, default=7)
    p.add_argument("--crop_top", type=int, default=0)
    p.add_argument("--roi_mask", type=str, default="")

    # Temporal persistence (pixel-level)
    p.add_argument("--persist_k", type=int, default=4, help="Temporal window size for mask persistence")
    p.add_argument("--persist_m", type=int, default=3, help="Min frames in window to keep a pixel")

    # Box-level hysteresis tracker (anti-flicker)
    p.add_argument("--min_track_age", type=int, default=3, help="Stable track age before showing a box")
    p.add_argument("--max_missed", type=int, default=3, help="Max missed frames before dropping a track")
    p.add_argument("--iou_match_thr", type=float, default=0.3, help="IoU threshold for box matching in tracker")

    # Snapshots
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--snap_dir", type=str, default="detections")
    p.add_argument("--snap_min_disp", type=float, default=1.5)
    p.add_argument("--snap_min_cc", type=float, default=0.6)
    p.add_argument("--snap_min_z", type=float, default=0.0)
    p.add_argument("--snap_max_z", type=float, default=1e6)
    p.add_argument("--snap_cooldown", type=float, default=1.5)
    p.add_argument("--snap_pad", type=int, default=4)
    p.add_argument("--snap_debug", action="store_true")

    return p


def args_to_config(args) -> AppConfig:
    smd = SMDParams(
        y_eps=args.y_eps, dmin=args.dmin, dmax=args.dmax,
        min_area=args.min_area, max_area=args.max_area,
        thr_fast=args.thr_fast, thr_slow=args.thr_slow,
        min_flow=args.min_flow, min_flow_small=args.min_flow_small,
        stereo_patch=args.sterio_patch if hasattr(args, "sterio_patch") else args.stereo_patch,
        stereo_search_pad=args.stereo_search_pad,
        stereo_ncc_min=args.stereo_ncc_min,
        use_clahe=True, size_aware_morph=True,
    )

    extras = dict(
        night_auto=args.night_auto,
        night_luma_thr=args.night_luma_thr,
        min_area_night_mult=args.min_area_night_mult,
        noise_k_fast=args.noise_k_fast,
        noise_k_slow=args.noise_k_slow,
        morph_open_day=args.morph_open_day,
        morph_open_night=args.morph_open_night,
        crop_top=args.crop_top,
        roi_mask=args.roi_mask,
        persist_k=args.persist_k,
        persist_m=args.persist_m,
        track_min_age=args.min_track_age,
        track_max_missed=args.max_missed,
        track_iou_thr=args.iou_match_thr,
    )
    for k, v in extras.items():
        setattr(smd, k, v)

    return AppConfig(
        left=args.left, right=args.right, calib_dir=args.calib_dir,
        intrinsics=args.intrinsics, baseline=args.baseline,
        width=None if args.native_resolution else args.width,
        height=None if args.native_resolution else args.height,
        native_resolution=args.native_resolution,
        display=args.display, headless=args.headless, save_vis=args.save_vis,
        smd=smd,
        reader=args.reader, ffmpeg=args.ffmpeg, mjpeg_q=args.mjpeg_q, ff_threads=args.ff_threads,
        display_max_w=args.display_max_w, display_max_h=args.display_max_h,
        snapshots=args.snapshots, snap_dir=args.snap_dir, snap_min_disp=args.snap_min_disp,
        snap_min_cc=args.snap_min_cc, snap_min_z=args.snap_min_z, snap_max_z=args.snap_max_z,
        snap_cooldown=args.snap_cooldown, snap_pad=args.snap_pad, snap_debug=args.snap_debug,
    )


# ------------------------------- PIPELINE ---------------------------------

def run(cfg: AppConfig):
    print("[*] Opening streams (opencv/ffmpeg)...")
    L = open_stream(cfg.left, cfg.width or 0, cfg.height or 0,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)
    R = open_stream(cfg.right, cfg.width or 0, cfg.height or 0,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)

    okL, fL, tL = warmup(L, "LEFT",  timeout_s=8.0)
    okR, fR, tR = warmup(R, "RIGHT", timeout_s=8.0)

    # Fallback: ffmpeg_mjpeg -> opencv, если кадры не пошли
    if cfg.reader == "ffmpeg_mjpeg":
        if not okL:
            print("[fallback] LEFT: switching to OpenCV reader")
            try: L.release()
            except Exception: pass
            L = open_stream(cfg.left, cfg.width or 0, cfg.height or 0,
                            reader="opencv", ffmpeg=cfg.ffmpeg)
            okL, fL, tL = warmup(L, "LEFT-OpenCV", timeout_s=5.0)
        if not okR:
            print("[fallback] RIGHT: switching to OpenCV reader")
            try: R.release()
            except Exception: pass
            R = open_stream(cfg.right, cfg.width or 0, cfg.height or 0,
                            reader="opencv", ffmpeg=cfg.ffmpeg)
            okR, fR, tR = warmup(R, "RIGHT-OpenCV", timeout_s=5.0)

    if not (okL and okR):
        raise RuntimeError("Failed to read initial frames from both cameras.")

    h, w = fL.shape[:2]
    frame_size = (w, h)

    calib = load_calibration(Path(cfg.calib_dir) if cfg.calib_dir else None,
                             Path(cfg.intrinsics) if cfg.intrinsics else None,
                             frame_size,
                             baseline_m=cfg.baseline)

    det = StereoMotionDetector(frame_size, cfg.smd)
    bufL, bufR = deque(maxlen=5), deque(maxlen=5)
    bufL.append((tL, fL)); bufR.append((tR, fR))

    writer = None
    if cfg.save_vis:
        writer = make_writer(cfg.save_vis, (w * 2, h), fps=20.0)

    saver = None
    if cfg.snapshots:
        saver = SnapshotSaver(cfg.snap_dir, cfg.snap_min_disp, cfg.snap_min_cc,
                              cfg.snap_min_z, cfg.snap_max_z, cfg.snap_cooldown, cfg.snap_pad,
                              debug=cfg.snap_debug)
        print("[snap_dir] ->", Path(cfg.snap_dir).resolve())

    # Box-level трекеры (анти-мерцание)
    trL = BoxTracker(iou_thr=float(getattr(cfg.smd, "track_iou_thr", 0.3)),
                     min_age=int(getattr(cfg.smd, "track_min_age", 3)),
                     max_missed=int(getattr(cfg.smd, "track_max_missed", 3)))
    trR = BoxTracker(iou_thr=float(getattr(cfg.smd, "track_iou_thr", 0.3)),
                     min_age=int(getattr(cfg.smd, "track_min_age", 3)),
                     max_missed=int(getattr(cfg.smd, "track_max_missed", 3)))

    show = cfg.display and not cfg.headless
    print("[*] Running. Press ESC to stop.")
    try:
        while True:
            okL, fL, tL = L.read()
            okR, fR, tR = R.read()
            if okL: bufL.append((tL, fL))
            if okR: bufR.append((tR, fR))
            if not bufL or not bufR:
                continue

            frameL, frameR = best_time_aligned(bufL, bufR)
            rL, rR = rectified_pair(calib, frameL, frameR)

            mL, mR, boxesL, boxesR, pairs = det.step(rL, rR)

            # --- Hysteresis tracking to suppress flicker ---
            keepL = trL.update(boxesL)
            keepR = trR.update(boxesR)
            if keepL or keepR:
                def _filter_boxes(boxes, keep):
                    new_boxes = []
                    remap = {}
                    for i, b in enumerate(boxes):
                        if i in keep:
                            remap[i] = len(new_boxes)
                            new_boxes.append(b)
                    return new_boxes, remap
                boxesL, mapL = _filter_boxes(boxesL, keepL)
                boxesR, mapR = _filter_boxes(boxesR, keepR)
                new_pairs = []
                for (i, j) in pairs:
                    if i in mapL and j in mapR:
                        new_pairs.append((mapL[i], mapR[j]))
                pairs = new_pairs
            else:
                boxesL, boxesR, pairs = [], [], []

            # --- Visualization ---
            visL = rL.copy(); visR = rR.copy()
            draw_boxes(visL, boxesL, (0, 255, 0), "L")
            draw_boxes(visR, boxesR, (255, 0, 0), "R")
            vis = stack_lr(visL, visR)

            # --- Diagnostics overlay ---
            try:
                meanY = int(np.mean(cv.cvtColor(rL, cv.COLOR_BGR2GRAY)))
            except Exception:
                meanY = -1
            thr_fast_val = getattr(det.params, 'thr_fast', None)
            thr_slow_val = getattr(det.params, 'thr_slow', None)
            thr_fast_str = f"{float(thr_fast_val):.1f}" if thr_fast_val is not None else "auto"
            thr_slow_str = f"{float(thr_slow_val):.1f}" if thr_slow_val is not None else "auto"
            min_area_str = str(getattr(det.params, 'min_area', 0))
            txt = (f"NIGHT={getattr(det, 'is_night', False)}  "
                   f"Y={meanY}  "
                   f"thr_fast={thr_fast_str}  "
                   f"thr_slow={thr_slow_str}  "
                   f"min_area={min_area_str}")
            cv.putText(vis, txt, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

            if writer is not None:
                writer.write(vis)

            if saver is not None:
                for (i, j) in pairs:
                    bl = boxesL[i]; br = boxesR[j]
                    cLx, cLy = int(bl.cx), int(bl.cy)
                    cRx, cRy = int(br.cx), int(br.cy)
                    disp = float(cLx - cRx)
                    Q = getattr(calib, "Q", None) if hasattr(calib, "Q") else None
                    saver.maybe_save(rL, rR,
                                     (int(bl.x), int(bl.y), int(bl.w), int(bl.h)),
                                     (int(br.x), int(br.y), int(br.w), int(br.h)),
                                     disp, Q=Q)

            if show:
                imshow_resized("VisualRadar L|R", vis, cfg.display_max_w, cfg.display_max_h)
                if (cv.waitKey(1) & 0xFF) == 27:
                    break
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        try:
            L.release(); R.release()
        except Exception:
            pass
        if show:
            cv.destroyAllWindows()


# ------------------------------- ENTRY ------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)

if __name__ == "__main__":
    main()
