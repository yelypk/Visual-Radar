from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv

from .config import AppConfig, SMDParams
from .calibration import load_calibration, rectified_pair
from .detector import StereoMotionDetector
from .tracks import BoxTracker
from .snapshots import SnapshotSaver
from .visualize import draw_boxes, draw_hud, stack_lr, imshow_resized
from .io import open_stream, make_writer
from .utils import BBox

@dataclass
class ProcessOut:
    maskL: np.ndarray
    maskR: np.ndarray
    boxesL: List[BBox]
    boxesR: List[BBox]
    pairs: List[Tuple[int, int]]
    show_idx: List[int]
    rectL: np.ndarray
    rectR: np.ndarray

def build_runtime(cfg: AppConfig):
    L = open_stream(cfg.left, cfg.width, cfg.height, cfg.reader, cfg.ffmpeg,
                    cfg.mjpeg_q, cfg.ff_threads, cfg.cap_buffersize, cfg.read_timeout)
    R = open_stream(cfg.right, cfg.width, cfg.height, cfg.reader, cfg.ffmpeg,
                    cfg.mjpeg_q, cfg.ff_threads, cfg.cap_buffersize, cfg.read_timeout)
    calib = load_calibration(cfg.calib_dir, cfg.intrinsics, (cfg.width, cfg.height), cfg.baseline)

    detector = StereoMotionDetector((cfg.width, cfg.height), cfg.smd)
    tracker = BoxTracker(cfg.iou_thr, cfg.min_age, cfg.max_missed, cfg.min_disp_pair, cfg.min_speed)
    snapper = SnapshotSaver(cfg.snap_dir, cfg.snap_min_disp, cfg.snap_min_cc,
                            cfg.snap_cooldown, cfg.snap_pad, cfg.snap_debug) if cfg.snapshots else None
    writer = make_writer(cfg.save_path, (2*cfg.width, cfg.height), cfg.save_fps, fourcc="mp4v", is_color=True) if cfg.save_vis else None
    return L, R, calib, detector, tracker, snapper, writer

def process_pair(cfg: AppConfig, detector: StereoMotionDetector, tracker: BoxTracker,
                 calib, frameL: np.ndarray, frameR: np.ndarray) -> ProcessOut:
    rectL, rectR = rectified_pair(calib, frameL, frameR) 
    mL, mR, boxesL, boxesR, pairs = detector.step(rectL, rectR)

    show_idx = tracker.update([boxesL[i] for i, _ in pairs])
    return ProcessOut(mL, mR, boxesL, boxesR, pairs, show_idx, rectL, rectR)

def render_and_record(cfg: AppConfig, proc: ProcessOut, tracker: BoxTracker,
                      writer, fps_txt: Optional[str] = None) -> np.ndarray:
    visL = proc.rectL.copy()
    visR = proc.rectR.copy()

    draw_boxes(visL, proc.boxesL, (0, 255, 0), "L")
    draw_boxes(visR, proc.boxesR, (255, 0, 0), "R")

    for k in proc.show_idx:
        iL, iR = proc.pairs[k]
        draw_boxes(visL, [proc.boxesL[iL]], (0, 255, 255), "trk")
        draw_boxes(visR, [proc.boxesR[iR]], (0, 255, 255), "trk")
    vis = stack_lr(visL, visR)
    hud_lines = [f"pairs:{len(proc.pairs)} show:{len(proc.show_idx)}"]
    if fps_txt: hud_lines.append(fps_txt)
    draw_hud(vis, hud_lines, corner="tr", alpha=0.55)
    if cfg.display:
        imshow_resized("visual-radar", vis, cfg.display_max_w, cfg.display_max_h)
    if writer is not None:
        writer.write(vis)
    return vis
