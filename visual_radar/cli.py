from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Tuple, Set, Optional

import cv2 as cv
import numpy as np

from visual_radar.config import AppConfig, SMDParams
from visual_radar.calibration import load_calibration, rectified_pair
from visual_radar.io import open_stream, make_writer
from visual_radar.visualize import draw_boxes, stack_lr, imshow_resized, draw_hud
from visual_radar.snapshots import SnapshotSaver
from visual_radar.tracks import BoxTracker
from visual_radar.detector import StereoMotionDetector


def build_parser():
    import argparse
    parser = argparse.ArgumentParser("visual_radar")

    # Sources
    parser.add_argument("--left", required=True, help="RTSP/file/path (left)")
    parser.add_argument("--right", required=True, help="RTSP/file/path (right)")
    parser.add_argument("--reader", default="opencv", choices=["opencv", "ffmpeg_mjpeg"])
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--mjpeg_q", type=int, default=6)
    parser.add_argument("--ff_threads", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    # Calibration
    parser.add_argument("--calib_dir", type=str, default="stereo_rtsp_out")
    parser.add_argument("--intrinsics", type=str, default=None)
    parser.add_argument("--baseline", type=float, default=None)

    # Output
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display_max_w", type=int, default=1920)
    parser.add_argument("--display_max_h", type=int, default=1080)
    parser.add_argument("--window", choices=["normal", "autosize"], default="normal")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_path", type=str, default="out.mp4")
    parser.add_argument("--save_fps", type=float, default=15.0)

    # Night/Day
    parser.add_argument("--night_auto", action="store_true")
    parser.add_argument("--night_luma_thr", type=float, default=50.0)
    parser.add_argument("--min_area_night_mult", type=float, default=4.0)
    parser.add_argument("--morph_open_night", type=int, default=7)
    parser.add_argument("--morph_open_day", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=25)
    parser.add_argument("--max_area", type=int, default=0)

    # Noise/Threshold
    parser.add_argument("--noise_k_fast", type=float, default=2.0)
    parser.add_argument("--noise_k_slow", type=float, default=1.0)

    # Persistence
    parser.add_argument("--persist_k", type=int, default=4)
    parser.add_argument("--persist_m", type=int, default=3)

    # ROI/Crop
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--roi_mask", type=str, default=None)

    # Drift
    parser.add_argument("--drift_max_fg_pct", type=float, default=0.25)
    parser.add_argument("--drift_max_frames", type=int, default=60)

    # Tracker
    parser.add_argument("--min_track_age", type=int, default=3)
    parser.add_argument("--max_missed", type=int, default=3)
    parser.add_argument("--iou_match_thr", type=float, default=0.3)
    parser.add_argument("--min_track_disp", type=float, default=6.0)
    parser.add_argument("--min_track_speed", type=float, default=1.2)

    # Stereo/top of frame
    parser.add_argument("--min_disp_pair", type=float, default=2.5)
    parser.add_argument("--y_area_split", type=float, default=0.55)

    # Snapshots
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--snap_dir", type=str, default="detections")
    parser.add_argument("--snap_min_cc", type=float, default=0.6)
    parser.add_argument("--snap_min_disp", type=float, default=1.5)
    parser.add_argument("--snap_pad", type=int, default=4)
    parser.add_argument("--snap_cooldown", type=float, default=1.5)
    parser.add_argument("--snap_debug", action="store_true")

    # Synchronization
    parser.add_argument("--sync_max_dt", type=float, default=0.05)
    parser.add_argument("--sync_attempts", type=int, default=4)

    # Stream health
    parser.add_argument("--cap_buffersize", type=int, default=1)
    parser.add_argument("--stall_timeout", type=float, default=3.0)
    parser.add_argument("--max_consec_fail", type=int, default=30)

    # Backend/system
    parser.add_argument("--cv_threads", type=int, default=0)
    parser.add_argument("--no_optimized", action="store_true")
    parser.add_argument("--print_fps", action="store_true")

    # Artifact filter
    parser.add_argument("--no_drop_artifacts", action="store_true")

    return parser


def args_to_config(args) -> AppConfig:
    smd = SMDParams(
        night_auto=bool(args.night_auto),
        night_luma_thr=float(args.night_luma_thr),
        min_area=int(args.min_area),
        max_area=int(args.max_area),
        min_area_night_mult=float(args.min_area_night_mult),
        morph_open_day=int(args.morph_open_day),
        morph_open_night=int(args.morph_open_night),
        thr_fast=float(args.noise_k_fast),
        thr_slow=float(args.noise_k_slow),
        crop_top=int(args.crop_top),
        roi_mask=args.roi_mask,
        persist_k=int(args.persist_k),
        persist_m=int(args.persist_m),
        drift_max_fg_pct=float(args.drift_max_fg_pct),
        drift_max_frames=int(args.drift_max_frames),
        track_iou_thr=float(args.iou_match_thr),
        track_min_age=int(args.min_track_age),
        track_max_missed=int(args.max_missed),
        track_min_disp=float(args.min_track_disp),
        track_min_speed=float(args.min_track_speed),
        min_disp_pair=float(args.min_disp_pair),
        y_area_split=float(args.y_area_split),
        sails_only=False,
    )

    return AppConfig(
        left=args.left, right=args.right,
        calib_dir=args.calib_dir, intrinsics=args.intrinsics, baseline=args.baseline,
        width=int(args.width), height=int(args.height),
        display=bool(args.display), save_vis=bool(args.save_vis),
        smd=smd,
        reader=args.reader, ffmpeg=args.ffmpeg, mjpeg_q=int(args.mjpeg_q), ff_threads=int(args.ff_threads),
        display_max_w=int(args.display_max_w), display_max_h=int(args.display_max_h),
        window=args.window,
        snapshots=bool(args.snapshots), snap_dir=args.snap_dir,
        snap_min_cc=float(args.snap_min_cc), snap_min_disp=float(args.snap_min_disp),
        snap_pad=int(args.snap_pad), snap_cooldown=float(args.snap_cooldown), snap_debug=bool(args.snap_debug),
        save_path=args.save_path, save_fps=float(args.save_fps),
        sync_max_dt=float(args.sync_max_dt), sync_attempts=int(args.sync_attempts),
        cap_buffersize=int(args.cap_buffersize),
        stall_timeout=float(args.stall_timeout), max_consec_fail=int(args.max_consec_fail),
        cv_threads=int(args.cv_threads), use_optimized=not bool(args.no_optimized),
        print_fps=bool(args.print_fps),
        drop_artifacts=not bool(args.no_drop_artifacts),
    )

# --- FFmpeg debug helpers -----------------------------------------------------

def _ff_tail(reader) -> str:
    """
    Вернёт последние строки stderr от ffmpeg, если ридер это поддерживает.
    Безопасно: если метода нет — вернёт пустую строку.
    """
    try:
        if hasattr(reader, "last_stderr"):
            return reader.last_stderr() or ""
    except Exception as e:
        return f"<stderr read error: {e!r}>"
    return ""

def _ff_state(tag: str, reader) -> None:
    """
    Сдампить полезную диагностику по процессу и stderr.
    """
    rc = None
    try:
        proc = getattr(reader, "proc", None)
        if proc is not None:
            rc = proc.poll()
    except Exception:
        pass

    tail = _ff_tail(reader)
    if rc is not None:
        logging.warning("[ffmpeg:%s] rc=%s", tag, rc)
    if tail:
        logging.warning("[ffmpeg:%s] stderr tail:\n%s", tag, tail)


def is_artifact_frame(bgr: np.ndarray) -> bool:
    """
    Heuristic for detecting corrupted frames:
    - Take the lower 40% of the frame, compute std by columns;
    - Excessive vertical dispersion indicates stripes/packet loss;
    - Additional trigger: nearly constant frame (very low std).
    """
    if bgr is None or bgr.size == 0:
        return True
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    band = gray[int(h * 0.6):, :]
    if band.size == 0:
        return False
    sigma_all = float(band.std())
    if sigma_all < 1.0:
        return True
    col_std = band.std(axis=0)
    if float(np.median(col_std)) > (sigma_all * 2.5):
        return True
    return False


def warmup(reader, timeout_s: float = 20.0) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
    """
    Try to read a valid frame from the stream within timeout.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, frame, ts = reader.read()
        if ok and frame is not None:
            return True, frame, ts
        time.sleep(0.05)
    return False, None, None


def sync_pair(L, R, dt_max: float, attempts: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Active synchronization: if |dt| > dt_max, read the lagging side up to 'attempts' times to quickly align streams.
    """
    okL, frameL, tL = L.read()
    okR, frameR, tR = R.read()
    if not okL or not okR:
        return False, None, None, None, None
    tries = 0
    while abs(float(tL) - float(tR)) > dt_max and tries < attempts:
        if tL < tR:
            okL, frameL, tL = L.read()
            if not okL:
                return False, None, None, None, None
        else:
            okR, frameR, tR = R.read()
            if not okR:
                return False, None, None, None, None
        tries += 1
    if abs(float(tL) - float(tR)) > dt_max:
        return False, None, None, None, None
    return True, frameL, frameR, tL, tR


def run(cfg: AppConfig) -> None:
    """
    Main loop for stereo motion detection and visualization.
    """
    # Global OpenCV settings
    if cfg.cv_threads and cfg.cv_threads > 0:
        cv.setNumThreads(int(cfg.cv_threads))
    cv.setUseOptimized(bool(cfg.use_optimized))

    L = open_stream(cfg.left, cfg.width, cfg.height,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads,
                    cap_buffersize=cfg.cap_buffersize)
    R = open_stream(cfg.right, cfg.width, cfg.height,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads,
                    cap_buffersize=cfg.cap_buffersize)
    logging.info("Reader LEFT : %s.%s", L.__class__.__module__, L.__class__.__name__)
    logging.info("Reader RIGHT: %s.%s", R.__class__.__module__, R.__class__.__name__)

    okL, frameL0, _ = warmup(L, timeout_s=20.0)
    okR, frameR0, _ = warmup(R, timeout_s=20.0)

    if not okL or not okR:
        # <<< добавили диагностику перед остановкой >>>
        _ff_state("L", L)
        _ff_state("R", R)
        logging.error("[warmup] No frames within 20s. Stopping.")
        try:
            L.release()
            R.release()
        except Exception:
            pass
        return

    # Synchronize sizes with actual frames
    h0, w0 = frameL0.shape[:2]
    cfg.width, cfg.height = w0, h0

    # Window
    if cfg.display:
        flag = cv.WINDOW_NORMAL if cfg.window == "normal" else cv.WINDOW_AUTOSIZE
        cv.namedWindow("VisualRadar L|R", flag)

    # Calibration/remap
    calib = load_calibration(
        Path(cfg.calib_dir),
        Path(cfg.intrinsics) if cfg.intrinsics else None,
        (cfg.width, cfg.height),
        cfg.baseline,
    )

    frame_size: Tuple[int, int] = (cfg.width, cfg.height)
    detector = StereoMotionDetector(frame_size, cfg.smd)

    trackerL = BoxTracker(
        iou_thr=float(cfg.smd.track_iou_thr),
        min_age=int(cfg.smd.track_min_age),
        max_missed=int(cfg.smd.track_max_missed),
        min_disp=float(cfg.smd.track_min_disp),
        min_speed=float(cfg.smd.track_min_speed),
    )
    trackerR = BoxTracker(
        iou_thr=float(cfg.smd.track_iou_thr),
        min_age=int(cfg.smd.track_min_age),
        max_missed=int(cfg.smd.track_max_missed),
        min_disp=float(cfg.smd.track_min_disp),
        min_speed=float(cfg.smd.track_min_speed),
    )

    snapshot_saver: Optional[SnapshotSaver] = None
    if cfg.snapshots:
        snapshot_saver = SnapshotSaver(
            out_dir=cfg.snap_dir,
            min_disp=cfg.snap_min_disp,
            min_cc=cfg.snap_min_cc,
            cooldown=cfg.snap_cooldown,
            pad=cfg.snap_pad,
            debug=cfg.snap_debug,
        )

    writer = None
    save_path = Path(cfg.save_path) if cfg.save_vis else None

    # Health/reconnect
    last_ok = time.time()
    failL = failR = 0

    # FPS print
    t_fps = time.perf_counter()
    fps_cnt = 0
    loop_fps = 0.0

    logging.info("[*] Running. Press ESC to stop.")
    try:
        while True:
            ok, frameL, frameR, tL, tR = sync_pair(L, R, cfg.sync_max_dt, cfg.sync_attempts)
            if not ok:
                # Count individual stream failures
                okl, _, _ = L.read()
                okr, _, _ = R.read()
                failL += int(not okl)
                failR += int(not okr)

                # <<< добавили: «ступеньки» для логов при затяжной тишине >>>
                if failL in (5, 20, 100):
                    logging.warning("[left] no frames, failL=%d", failL)
                    _ff_state("L", L)
                if failR in (5, 20, 100):
                    logging.warning("[right] no frames, failR=%d", failR)
                    _ff_state("R", R)
            else:
                # Artifact frame filter
                if cfg.drop_artifacts and (is_artifact_frame(frameL) or is_artifact_frame(frameR)):
                    continue

                failL = failR = 0
                last_ok = time.time()

                rectL, rectR = rectified_pair(calib, frameL, frameR)
                _, _, boxesL, boxesR, pairs = detector.step(rectL, rectR)

                keepL: Set[int] = trackerL.update(boxesL)
                keepR: Set[int] = trackerR.update(boxesR)
                boxesL = [b for i, b in enumerate(boxesL) if i in keepL]
                boxesR = [b for i, b in enumerate(boxesR) if i in keepR]

                visL = rectL.copy()
                visR = rectR.copy()
                draw_boxes(visL, boxesL, (0, 255, 0), "L")
                draw_boxes(visR, boxesR, (255, 0, 0), "R")
                vis = stack_lr(visL, visR)

                # HUD
                dt_ms = int(abs(float(tL) - float(tR)) * 1000.0)
                hud_lines = [
                    f"dt: {dt_ms} ms  (thr {int(cfg.sync_max_dt*1000)} ms)",
                    f"night: {detector.is_night}",
                    f"fails L/R: {failL}/{failR}",
                    f"fps: {loop_fps:.1f}",
                ]
                draw_hud(vis, hud_lines, corner="tr")

                if cfg.save_vis:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = make_writer(str(save_path), (w, h), fps=cfg.save_fps)
                    writer.write(vis)

                if cfg.display:
                    imshow_resized("VisualRadar L|R", vis, cfg.display_max_w, cfg.display_max_h)
                    if (cv.waitKey(1) & 0xFF) == 27:
                        break

                if snapshot_saver is not None and pairs:
                    Q = getattr(calib, "Q", None) if hasattr(calib, "Q") else None
                    for (i, j) in pairs:
                        if i >= len(boxesL) or j >= len(boxesR):
                            continue
                        bl, br = boxesL[i], boxesR[j]
                        disp = float(bl.cx - br.cx)
                        snapshot_saver.maybe_save(
                            rectL, rectR,
                            (int(bl.x), int(bl.y), int(bl.w), int(bl.h)),
                            (int(br.x), int(br.y), int(br.w), int(br.h)),
                            disp, Q=Q,
                        )

                # FPS
                fps_cnt += 1
                if cfg.print_fps and (time.perf_counter() - t_fps) >= 1.0:
                    loop_fps = float(fps_cnt) / (time.perf_counter() - t_fps)
                    logging.info(f"[loop] {loop_fps:.1f} FPS")
                    fps_cnt = 0
                    t_fps = time.perf_counter()

            # Reconnect on silence or frequent failures
            if (time.time() - last_ok) > cfg.stall_timeout or failL >= cfg.max_consec_fail:
                logging.warning("[health] Reopen LEFT stream…")
                _ff_state("L", L)                 # <<< добавили
                try:
                    L.reopen()
                except Exception:
                    pass
                failL = 0
                last_ok = time.time()

            if (time.time() - last_ok) > cfg.stall_timeout or failR >= cfg.max_consec_fail:
                logging.warning("[health] Reopen RIGHT stream…")
                _ff_state("R", R)                 # <<< добавили
                try:
                    R.reopen()
                except Exception:
                    pass
                failR = 0
                last_ok = time.time()

    except KeyboardInterrupt:
        logging.info("[!] Interrupted by user.")
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            L.release()
            R.release()
        except Exception:
            pass
        if cfg.display:
            cv.destroyAllWindows()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()