import argparse
from collections import deque
from pathlib import Path
import cv2 as cv
import numpy as np

from .config import AppConfig, SMDParams
from .io import RTSPReader, make_writer
from .calibration import load_calibration, rectified_pair
from .detector import StereoMotionDetector
from .sync import best_time_aligned
from .visualize import draw_boxes, stack_lr

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
    p.add_argument("--display", action="store_true", help="Show debug windows")
    p.add_argument("--headless", action="store_true", help="Disable any GUI display")
    p.add_argument("--save_vis", type=str, default=None, help="Path to save visualization video (mp4)")
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
        smd=smd
    )

def run(cfg: AppConfig):
    # Open streams
    L = RTSPReader(cfg.left, cfg.width, cfg.height)
    R = RTSPReader(cfg.right, cfg.width, cfg.height)
    okL,fL,tL = L.read(); okR,fR,tR = R.read()
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

    show = cfg.display and not cfg.headless
    print("[*] Running. Press Ctrl+C to stop.")
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

            if show:
                cv.imshow("VisualRadar L|R", vis)
                if (cv.waitKey(1) & 0xFF) == 27:
                    break
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        L.release(); R.release()
        if show:
            cv.destroyAllWindows()

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)

if __name__ == "__main__":
    main()
