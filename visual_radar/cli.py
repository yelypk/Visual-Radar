import argparse
from collections import deque
from pathlib import Path
import cv2 as cv
import numpy as np
import time

from .config import AppConfig, SMDParams
from .io import open_stream, make_writer
from .calibration import load_calibration, rectified_pair
from .detector import StereoMotionDetector
from .sync import best_time_aligned
from .visualize import draw_boxes, stack_lr, imshow_resized
from .snapshots import SnapshotSaver

def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--left", required=True, help="RTSP (or video) source for LEFT camera")
    p.add_argument("--right", required=True, help="RTSP (or video) source for RIGHT camera")
    p.add_argument("--calib_dir", type=str, default="stereo_rtsp_out", help="Dir with maps/Q.npy if available")
    p.add_argument("--intrinsics", type=str, default=None, help="Path to intrinsics .npz (K1,K2,D1,D2,R,T)")
    p.add_argument("--baseline", type=float, default=None, help="Baseline meters (if T not in intrinsics)")
    p.add_argument("--width", type=int, default=None, help="Force capture width (None = native)")
    p.add_argument("--height", type=int, default=None, help="Force capture height (None = native)")
    p.add_argument("--native_resolution", action="store_true", help="Prefer native stream resolution")
    p.add_argument("--small_targets", action="store_true", help="Lower thresholds for tiny targets")
    p.add_argument("--y_eps", type=int, default=16, help="Epipolar vertical tolerance (px)")
    p.add_argument("--dmin", type=int, default=1, help="Min valid disparity (px)")
    p.add_argument("--dmax", type=int, default=400, help="Max valid disparity (px)")
    p.add_argument("--min_area", type=int, default=25, help="Minimum bbox area in pixels")
    p.add_argument("--max_area", type=int, default=0, help="Maximum bbox area (0=disabled)")
    p.add_argument("--thr_fast", type=float, default=None, help="Override fast diff threshold (auto if None)")
    p.add_argument("--thr_slow", type=float, default=None, help="Override slow diff threshold (auto if None)")
    p.add_argument("--min_flow", type=float, default=0.25, help="Min flow magnitude for normal boxes")
    p.add_argument("--min_flow_small", type=float, default=0.01, help="Min flow magnitude for tiny boxes")
    p.add_argument("--stereo_patch", type=int, default=13, help="NCC template patch size (odd)")
    p.add_argument("--stereo_search_pad", type=int, default=60, help="Search radius in x (px) for NCC fallback")
    p.add_argument("--stereo_ncc_min", type=float, default=0.40, help="Min NCC to accept fallback match")
    p.add_argument("--reader", choices=["opencv","ffmpeg_mjpeg"], default="opencv", help="Video reader backend")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary path")
    p.add_argument("--mjpeg_q", type=int, default=6, help="MJPEG quality for ffmpeg_mjpeg (2..8; lower=better)")
    p.add_argument("--ff_threads", type=int, default=3, help="Threads for ffmpeg MJPEG")
    p.add_argument("--display", action="store_true", help="Show debug windows")
    p.add_argument("--display_max_w", type=int, default=1280)
    p.add_argument("--display_max_h", type=int, default=720)
    p.add_argument("--headless", action="store_true", help="Disable any GUI display")
    p.add_argument("--save_vis", type=str, default=None, help="Path to save visualization video (mp4)")
    p.add_argument("--snapshots", action="store_true", help="Enable saving snapshots of detections")
    p.add_argument("--snap_dir", type=str, default="detections")
    p.add_argument("--snap_min_disp", type=float, default=1.5)
    p.add_argument("--snap_min_cc", type=float, default=0.60)
    p.add_argument("--snap_min_z", type=float, default=0.0)
    p.add_argument("--snap_max_z", type=float, default=5000.0)
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
        stereo_patch=args.stereo_patch, stereo_search_pad=args.stereo_search_pad,
        stereo_ncc_min=args.stereo_ncc_min,
        use_clahe=True, size_aware_morph=True
    )
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
        snap_cooldown=args.snap_cooldown, snap_pad=args.snap_pad, snap_debug=args.snap_debug
    )

def warmup(reader, name, timeout_s=8.0):
    t0 = time.time()
    tries = 0
    while time.time() - t0 < timeout_s:
        ok, f, ts = reader.read()
        tries += 1
        if ok:
            print(f"[warmup] {name}: got first frame {f.shape} after {tries} reads")
            return True, f, ts
        time.sleep(0.05)
    print(f"[warmup] {name}: no frames within {timeout_s}s")
    return False, None, None

def run(cfg: AppConfig):
    # Open streams with diagnostics + warmup + fallback
    print(f"[*] Opening streams ({cfg.reader})...")
    L = open_stream(cfg.left, cfg.width or 0, cfg.height or 0, reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)
    R = open_stream(cfg.right, cfg.width or 0, cfg.height or 0, reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)

    okL, fL, tL = warmup(L, "LEFT",  timeout_s=8.0)
    okR, fR, tR = warmup(R, "RIGHT", timeout_s=8.0)

    # Авто-фоллбек с ffmpeg_mjpeg -> opencv
    if cfg.reader == "ffmpeg_mjpeg":
        if not okL:
            print("[fallback] LEFT: switching to OpenCV reader")
            try: L.release()
            except Exception: pass
            L = open_stream(cfg.left, cfg.width or 0, cfg.height or 0, reader="opencv", ffmpeg=cfg.ffmpeg)
            okL, fL, tL = warmup(L, "LEFT-OpenCV", timeout_s=5.0)
        if not okR:
            print("[fallback] RIGHT: switching to OpenCV reader")
            try: R.release()
            except Exception: pass
            R = open_stream(cfg.right, cfg.width or 0, cfg.height or 0, reader="opencv", ffmpeg=cfg.ffmpeg)
            okR, fR, tR = warmup(R, "RIGHT-OpenCV", timeout_s=5.0)

    if not (okL and okR):
        raise RuntimeError("Failed to read initial frames from both cameras.")

    h,w = fL.shape[:2]
    frame_size = (w,h)

    calib = load_calibration(Path(cfg.calib_dir) if cfg.calib_dir else None,
                             Path(cfg.intrinsics) if cfg.intrinsics else None,
                             frame_size,
                             baseline_m=cfg.baseline)

    det = StereoMotionDetector(frame_size, cfg.smd)
    bufL, bufR = deque(maxlen=5), deque(maxlen=5)
    bufL.append((tL, fL)); bufR.append((tR, fR))

    writer = None
    if cfg.save_vis:
        writer = make_writer(cfg.save_vis, (w*2, h), fps=20.0)

    saver = None
    if cfg.snapshots:
        saver = SnapshotSaver(cfg.snap_dir, cfg.snap_min_disp, cfg.snap_min_cc,
                               cfg.snap_min_z, cfg.snap_max_z, cfg.snap_cooldown, cfg.snap_pad,
                               debug=cfg.snap_debug)
        print('[snap_dir] ->', Path(cfg.snap_dir).resolve())

    show = cfg.display and not cfg.headless
    print("[*] Running. Press ESC to stop.")
    try:
        while True:
            okL,fL,tL = L.read()
            okR,fR,tR = R.read()
            if okL: bufL.append((tL,fL))
            if okR: bufR.append((tR,fR))
            if not bufL or not bufR:
                continue
            frameL, frameR = best_time_aligned(bufL, bufR)
            rL, rR = rectified_pair(calib, frameL, frameR)

            mL, mR, boxesL, boxesR, pairs = det.step(rL, rR)

            visL = rL.copy(); visR = rR.copy()
            draw_boxes(visL, boxesL, (0,255,0), "L")
            draw_boxes(visR, boxesR, (255,0,0), "R")
            vis = stack_lr(visL, visR)

            if writer is not None:
                writer.write(vis)

            if saver is not None:
                for (i,j) in pairs:
                    bl = boxesL[i]; br = boxesR[j]
                    cLx, cLy = int(bl.cx), int(bl.cy)
                    cRx, cRy = int(br.cx), int(br.cy)
                    disp = float(cLx - cRx)
                    Q = getattr(calib, "Q", None) if hasattr(calib, "Q") else None
                    saver.maybe_save(rL, rR, (int(bl.x),int(bl.y),int(bl.w),int(bl.h)),
                                     (int(br.x),int(br.y),int(br.w),int(br.h)), disp, Q=Q)

            if show:
                imshow_resized("VisualRadar L|R", vis, cfg.display_max_w, cfg.display_max_h)
                if (cv.waitKey(1) & 0xFF) == 27:
                    break
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        try: L.release(); R.release()
        except Exception: pass
        if show:
            cv.destroyAllWindows()

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)