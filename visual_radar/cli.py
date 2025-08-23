# visual_radar/cli.py
from __future__ import annotations

import time
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser():
    import argparse
    p = argparse.ArgumentParser("visual_radar")

    # источники
    p.add_argument("--left", required=True, help="RTSP/файл/путь (левый)")
    p.add_argument("--right", required=True, help="RTSP/файл/путь (правый)")
    p.add_argument("--reader", default="opencv", choices=["opencv", "ffmpeg_mjpeg"])
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--mjpeg_q", type=int, default=6)
    p.add_argument("--ff_threads", type=int, default=3)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    # калибровка
    p.add_argument("--calib_dir", type=str, default="stereo_rtsp_out")
    p.add_argument("--intrinsics", type=str, default=None)
    p.add_argument("--baseline", type=float, default=None)

    # вывод
    p.add_argument("--display", action="store_true")
    p.add_argument("--display_max_w", type=int, default=1920)
    p.add_argument("--display_max_h", type=int, default=1080)
    p.add_argument("--window", choices=["normal", "autosize"], default="normal")
    p.add_argument("--save_vis", action="store_true")
    p.add_argument("--save_path", type=str, default="out.mp4")
    p.add_argument("--save_fps", type=float, default=15.0)

    # день/ночь
    p.add_argument("--night_auto", action="store_true")
    p.add_argument("--night_luma_thr", type=float, default=50.0)
    p.add_argument("--min_area_night_mult", type=float, default=4.0)
    p.add_argument("--morph_open_night", type=int, default=7)

    # день
    p.add_argument("--morph_open_day", type=int, default=3)
    p.add_argument("--min_area", type=int, default=25)
    p.add_argument("--max_area", type=int, default=0)

    # шум/порог движения
    p.add_argument("--noise_k_fast", type=float, default=2.0)
    p.add_argument("--noise_k_slow", type=float, default=1.0)

    # персистентность
    p.add_argument("--persist_k", type=int, default=4)
    p.add_argument("--persist_m", type=int, default=3)

    # ROI/кроп
    p.add_argument("--crop_top", type=int, default=0)
    p.add_argument("--roi_mask", type=str, default=None)

    # анти-дрейф
    p.add_argument("--drift_max_fg_pct", type=float, default=0.25)
    p.add_argument("--drift_max_frames", type=int, default=60)

    # трекер
    p.add_argument("--min_track_age", type=int, default=3)
    p.add_argument("--max_missed", type=int, default=3)
    p.add_argument("--iou_match_thr", type=float, default=0.3)
    p.add_argument("--min_track_disp", type=float, default=6.0)
    p.add_argument("--min_track_speed", type=float, default=1.2)

    # stereo / верх кадра
    p.add_argument("--min_disp_pair", type=float, default=2.5)
    p.add_argument("--y_area_split", type=float, default=0.55)

    # снапшоты
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--snap_dir", type=str, default="detections")
    p.add_argument("--snap_min_cc", type=float, default=0.6)
    p.add_argument("--snap_min_disp", type=float, default=1.5)
    p.add_argument("--snap_pad", type=int, default=4)
    p.add_argument("--snap_cooldown", type=float, default=1.5)
    p.add_argument("--snap_debug", action="store_true")

    # синхронизация
    p.add_argument("--sync_max_dt", type=float, default=0.05)
    p.add_argument("--sync_attempts", type=int, default=4)

    # здоровье потоков
    p.add_argument("--cap_buffersize", type=int, default=1)
    p.add_argument("--stall_timeout", type=float, default=3.0)
    p.add_argument("--max_consec_fail", type=int, default=30)

    # системные/бэкенд
    p.add_argument("--cv_threads", type=int, default=0)
    p.add_argument("--no_optimized", action="store_true")
    p.add_argument("--print_fps", action="store_true")

    # фильтр «битых» кадров
    p.add_argument("--no_drop_artifacts", action="store_true")

    return p


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


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _is_artifact_frame(bgr: np.ndarray) -> bool:
    """
    Грубая эвристика «битого» кадра:
    - берём нижние 40% кадра, считаем std по колонкам;
    - чрезмерная вертикальная дисперсия → полосы/обрыв пакетов;
    - дополнительный триггер — почти константный кадр (очень малая std).
    """
    if bgr is None or bgr.size == 0:
        return True
    g = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    band = g[int(h * 0.6):, :]
    if band.size == 0:
        return False
    sigma_all = float(band.std())
    if sigma_all < 1.0:
        return True
    col_std = band.std(axis=0)
    if float(np.median(col_std)) > (sigma_all * 2.5):
        return True
    return False


def warmup(reader, timeout_s: float = 20.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, fr, ts = reader.read()
        if ok and fr is not None:
            return True, fr, ts
        time.sleep(0.05)
    return False, None, None


def _sync_pair(L, R, dt_max: float, attempts: int):
    """
    Активная синхронизация: если |dt|>dt_max — дочитываем «отстающую» сторону
    до attempts раз, чтобы быстро выровнять потоки.
    """
    okL, fL, tL = L.read()
    okR, fR, tR = R.read()
    if not okL or not okR:
        return False, None, None, None, None
    tries = 0
    while abs(float(tL) - float(tR)) > dt_max and tries < attempts:
        if tL < tR:
            okL, fL, tL = L.read()
            if not okL:
                return False, None, None, None, None
        else:
            okR, fR, tR = R.read()
            if not okR:
                return False, None, None, None, None
        tries += 1
    if abs(float(tL) - float(tR)) > dt_max:
        return False, None, None, None, None
    return True, fL, fR, tL, tR


# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------
def run(cfg: AppConfig):
    # глобальные настройки OpenCV
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

    okL, fL0, _ = warmup(L, timeout_s=20.0)
    okR, fR0, _ = warmup(R, timeout_s=20.0)

    if not okL or not okR:
        print("[warmup] no frames within 20s. stop.")
        try:
            L.release(); R.release()
        except Exception:
            pass
        return

    # синхронизируем размеры с реальными
    h0, w0 = fL0.shape[:2]
    cfg.width, cfg.height = w0, h0

    # окно
    if cfg.display:
        flag = cv.WINDOW_NORMAL if cfg.window == "normal" else cv.WINDOW_AUTOSIZE
        cv.namedWindow("VisualRadar L|R", flag)

    # калибровка/ремап
    calib = load_calibration(
        Path(cfg.calib_dir),
        Path(cfg.intrinsics) if cfg.intrinsics else None,
        (cfg.width, cfg.height),
        cfg.baseline,
    )

    frame_size: Tuple[int, int] = (cfg.width, cfg.height)
    det = StereoMotionDetector(frame_size, cfg.smd)

    trL = BoxTracker(
        iou_thr=float(cfg.smd.track_iou_thr),
        min_age=int(cfg.smd.track_min_age),
        max_missed=int(cfg.smd.track_max_missed),
        min_disp=float(cfg.smd.track_min_disp),
        min_speed=float(cfg.smd.track_min_speed),
    )
    trR = BoxTracker(
        iou_thr=float(cfg.smd.track_iou_thr),
        min_age=int(cfg.smd.track_min_age),
        max_missed=int(cfg.smd.track_max_missed),
        min_disp=float(cfg.smd.track_min_disp),
        min_speed=float(cfg.smd.track_min_speed),
    )

    saver: Optional[SnapshotSaver] = None
    if cfg.snapshots:
        saver = SnapshotSaver(
            out_dir=cfg.snap_dir,
            min_disp=cfg.snap_min_disp,
            min_cc=cfg.snap_min_cc,
            cooldown=cfg.snap_cooldown,
            pad=cfg.snap_pad,
            debug=cfg.snap_debug,
        )

    writer = None
    sp = Path(cfg.save_path) if cfg.save_vis else None

    # здоровье/переподключение
    last_ok = time.time()
    failL = failR = 0

    # FPS печать
    t_fps = time.perf_counter()
    fps_cnt = 0
    loop_fps = 0.0

    print("[*] Running.  Press ESC to stop.")
    try:
        while True:
            ok, fL, fR, tL, tR = _sync_pair(L, R, cfg.sync_max_dt, cfg.sync_attempts)
            if not ok:
                # грубо считаем индивидуальные фейлы источников
                okl, _, _ = L.read()
                okr, _, _ = R.read()
                failL += int(not okl)
                failR += int(not okr)
            else:
                # фильтр «битых» кадров
                if cfg.drop_artifacts and (_is_artifact_frame(fL) or _is_artifact_frame(fR)):
                    continue

                failL = failR = 0
                last_ok = time.time()

                rL, rR = rectified_pair(calib, fL, fR)
                _mL, _mR, boxesL, boxesR, pairs = det.step(rL, rR)

                keepL: Set[int] = trL.update(boxesL)
                keepR: Set[int] = trR.update(boxesR)
                boxesL = [b for i, b in enumerate(boxesL) if i in keepL]
                boxesR = [b for i, b in enumerate(boxesR) if i in keepR]

                visL = rL.copy()
                visR = rR.copy()
                draw_boxes(visL, boxesL, (0, 255, 0), "L")
                draw_boxes(visR, boxesR, (255, 0, 0), "R")
                vis = stack_lr(visL, visR)

                # HUD
                dt_ms = int(abs(float(tL) - float(tR)) * 1000.0)
                hud_lines = [
                    f"dt: {dt_ms} ms  (thr {int(cfg.sync_max_dt*1000)} ms)",
                    f"night: {det.is_night}",
                    f"fails L/R: {failL}/{failR}",
                    f"fps: {loop_fps:.1f}",
                ]
                draw_hud(vis, hud_lines, corner="tr")

                if cfg.save_vis:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = make_writer(str(sp), (w, h), fps=cfg.save_fps)
                    writer.write(vis)

                if cfg.display:
                    imshow_resized("VisualRadar L|R", vis, cfg.display_max_w, cfg.display_max_h)
                    if (cv.waitKey(1) & 0xFF) == 27:
                        break

                if saver is not None and pairs:
                    Q = getattr(calib, "Q", None) if hasattr(calib, "Q") else None
                    for (i, j) in pairs:
                        if i >= len(boxesL) or j >= len(boxesR):
                            continue
                        bl, br = boxesL[i], boxesR[j]
                        disp = float(bl.cx - br.cx)
                        saver.maybe_save(
                            rL, rR,
                            (int(bl.x), int(bl.y), int(bl.w), int(bl.h)),
                            (int(br.x), int(br.y), int(br.w), int(br.h)),
                            disp, Q=Q,
                        )

                # FPS
                fps_cnt += 1
                if cfg.print_fps and (time.perf_counter() - t_fps) >= 1.0:
                    loop_fps = float(fps_cnt) / (time.perf_counter() - t_fps)
                    print(f"[loop] {loop_fps:.1f} FPS")
                    fps_cnt = 0
                    t_fps = time.perf_counter()

            # переподключение при тишине или частых фейлах
            if (time.time() - last_ok) > cfg.stall_timeout or failL >= cfg.max_consec_fail:
                print("[health] reopen LEFT stream…")
                try:
                    L.reopen()
                except Exception:
                    pass
                failL = 0
                last_ok = time.time()
            if (time.time() - last_ok) > cfg.stall_timeout or failR >= cfg.max_consec_fail:
                print("[health] reopen RIGHT stream…")
                try:
                    R.reopen()
                except Exception:
                    pass
                failR = 0
                last_ok = time.time()

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
