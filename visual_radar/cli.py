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
from visual_radar.visualize import draw_boxes, stack_lr, imshow_resized
from visual_radar.snapshots import SnapshotSaver
from visual_radar.tracks import BoxTracker
from visual_radar.detector import StereoMotionDetector


# -----------------------------
# CLI parsing
# -----------------------------
def build_parser():
    import argparse

    p = argparse.ArgumentParser("visual_radar")

    # источники
    p.add_argument("--left", required=True, help="RTSP/файл/путь (левый)")
    p.add_argument("--right", required=True, help="RTSP/файл/путь (правый)")
    p.add_argument("--reader", default="opencv", choices=["opencv", "ffmpeg_mjpeg"], help="способ чтения")
    p.add_argument("--ffmpeg", default="ffmpeg", help="команда ffmpeg (если нужна)")
    p.add_argument("--mjpeg_q", type=int, default=6, help="качество MJPEG (ffmpeg_mjpeg)")
    p.add_argument("--ff_threads", type=int, default=3, help="потоки ffmpeg")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    # калибровка/ремап
    p.add_argument("--calib_dir", type=str, default="stereo_rtsp_out")
    p.add_argument("--intrinsics", type=str, default=None, help="путь к intrinsics.npz (опционально)")
    p.add_argument("--baseline", type=float, default=None, help="база камер в метрах (опционально) — для совместимости")

    # вывод
    p.add_argument("--display", action="store_true")
    p.add_argument("--display_max_w", type=int, default=1920)
    p.add_argument("--display_max_h", type=int, default=1080)
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

    # персистентность масок
    p.add_argument("--persist_k", type=int, default=4)
    p.add_argument("--persist_m", type=int, default=3)

    # подрезка/ROI
    p.add_argument("--crop_top", type=int, default=0)
    p.add_argument("--roi_mask", type=str, default=None)

    # анти-дрейф
    p.add_argument("--drift_max_fg_pct", type=float, default=0.25)
    p.add_argument("--drift_max_frames", type=int, default=60)

    # трекер (анти-мерцание + движение)
    p.add_argument("--min_track_age", type=int, default=3)
    p.add_argument("--max_missed", type=int, default=3)
    p.add_argument("--iou_match_thr", type=float, default=0.3)
    p.add_argument("--min_track_disp", type=float, default=6.0, help="суммарное смещение (px) за окно")
    p.add_argument("--min_track_speed", type=float, default=1.2, help="средняя скорость (px/кадр) за окно")

    # stereo / эпиполярка
    p.add_argument("--min_disp_pair", type=float, default=2.5, help="минимальная |диспаритет| по центрам боксов")
    p.add_argument("--y_area_split", type=float, default=0.55, help="граница неба/воды по высоте (0..1)")

    # снапшоты
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--snap_dir", type=str, default="detections")
    p.add_argument("--snap_min_cc", type=float, default=0.6)
    p.add_argument("--snap_min_disp", type=float, default=1.5)
    p.add_argument("--snap_pad", type=int, default=4)
    p.add_argument("--snap_cooldown", type=float, default=1.5)
    p.add_argument("--snap_debug", action="store_true")

    # SAILS ONLY
    p.add_argument("--sails_only", action="store_true", help="Фильтр под паруса (если нет ROI воды).")
    p.add_argument("--sails_water_split", type=float, default=0.55)
    p.add_argument("--sails_white_delta", type=float, default=18.0)
    p.add_argument("--sails_min_h_over_w", type=float, default=0.85)
    p.add_argument("--sails_max_w", type=int, default=240)
    p.add_argument("--sails_min_h", type=int, default=6)

    return p


def args_to_config(args) -> AppConfig:
    smd = SMDParams(
        # день/ночь
        night_auto=bool(args.night_auto),
        night_luma_thr=float(args.night_luma_thr),
        min_area=int(args.min_area),
        max_area=int(args.max_area),
        min_area_night_mult=float(args.min_area_night_mult),
        morph_open_day=int(args.morph_open_day),
        morph_open_night=int(args.morph_open_night),

        # движение/шум
        thr_fast=float(args.noise_k_fast),
        thr_slow=float(args.noise_k_slow),

        # морфология/маски
        crop_top=int(args.crop_top),
        roi_mask=args.roi_mask,

        # персистентность
        persist_k=int(args.persist_k),
        persist_m=int(args.persist_m),

        # анти-дрейф
        drift_max_fg_pct=float(args.drift_max_fg_pct),
        drift_max_frames=int(args.drift_max_frames),

        # трекер
        track_iou_thr=float(args.iou_match_thr),
        track_min_age=int(args.min_track_age),
        track_max_missed=int(args.max_missed),
        track_min_disp=float(args.min_track_disp),
        track_min_speed=float(args.min_track_speed),

        # stereo / верх кадра
        min_disp_pair=float(args.min_disp_pair),
        y_area_split=float(args.y_area_split),

        # sails-only
        sails_only=bool(args.sails_only),
        sails_water_split=float(args.sails_water_split),
        sails_white_delta=float(args.sails_white_delta),
        sails_min_h_over_w=float(args.sails_min_h_over_w),
        sails_max_w=int(args.sails_max_w),
        sails_min_h=int(args.sails_min_h),
    )

    return AppConfig(
        left=args.left,
        right=args.right,
        calib_dir=args.calib_dir,
        intrinsics=args.intrinsics,
        baseline=args.baseline,
        width=args.width,
        height=args.height,
        native_resolution=False,

        display=bool(args.display),
        headless=not bool(args.display),
        save_vis=bool(args.save_vis),

        smd=smd,

        # reader
        reader=args.reader,
        ffmpeg=args.ffmpeg,
        mjpeg_q=int(args.mjpeg_q),
        ff_threads=int(args.ff_threads),

        # вывод
        display_max_w=int(args.display_max_w),
        display_max_h=int(args.display_max_h),

        # снапшоты
        snapshots=bool(args.snapshots),
        snap_dir=args.snap_dir,
        snap_min_cc=float(args.snap_min_cc),
        snap_min_disp=float(args.snap_min_disp),
        snap_pad=int(args.snap_pad),
        snap_cooldown=float(args.snap_cooldown),
        snap_debug=bool(args.snap_debug),

        # запись
        save_path=args.save_path,
        save_fps=float(args.save_fps),
    )


# -----------------------------
# helpers
# -----------------------------
def warmup(reader, name: str, timeout_s: float = 20.0):
    """Ждём первый кадр от источника до timeout_s секунд (по умолчанию 20с)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, fr, ts = reader.read()
        if ok:
            return True, fr, ts
        time.sleep(0.05)
    return False, None, None


# -----------------------------
# main run loop
# -----------------------------
def run(cfg: AppConfig):
    # открыть источники
    L = open_stream(cfg.left, cfg.width, cfg.height,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)
    R = open_stream(cfg.right, cfg.width, cfg.height,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)

    # тёплый старт 20с
    okL, fL0, _ = warmup(L, "LEFT", timeout_s=20.0)
    okR, fR0, _ = warmup(R, "RIGHT", timeout_s=20.0)

    # авто-фоллбэк: если MJPEG не дал кадр — пробуем OpenCV/H.264
    if (not okL or not okR) and cfg.reader == "ffmpeg_mjpeg":
        print("[warmup] no frames via ffmpeg_mjpeg → fallback to opencv reader...")
        try:
            L.release(); R.release()
        except Exception:
            pass
        L = open_stream(cfg.left, cfg.width, cfg.height, reader="opencv",
                        ffmpeg=cfg.ffmpeg, mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)
        R = open_stream(cfg.right, cfg.width, cfg.height, reader="opencv",
                        ffmpeg=cfg.ffmpeg, mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)
        okL, fL0, _ = warmup(L, "LEFT", timeout_s=20.0)
        okR, fR0, _ = warmup(R, "RIGHT", timeout_s=20.0)

    if not okL or not okR:
        print("[warmup] no frames within 20s. stop.")
        try:
            L.release(); R.release()
        except Exception:
            pass
        return

    # калибровка/ректификация
    calib = load_calibration(
        Path(cfg.calib_dir),
        Path(cfg.intrinsics) if cfg.intrinsics else None,
        (cfg.width, cfg.height),
        cfg.baseline,
    )

    # детектор и трекеры
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

    # снапшоты (опционально)
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

    # запись (опционально)
    writer = None
    if cfg.save_vis:
        sp = Path(cfg.save_path)
        writer = make_writer(str(sp), (cfg.width * 2, cfg.height), fps=cfg.save_fps)

    print("[*] Running.  Press ESC to stop.")

    try:
        while True:
            okL, fL, tL = L.read()
            okR, fR, tR = R.read()
            if not okL or not okR:
                continue

            # ректификация
            rL, rR = rectified_pair(calib, fL, fR)

            # детекция
            mL, mR, boxesL, boxesR, pairs = det.step(rL, rR)

            # --- Stereo parallax gate: минимальный диспаритет по центрам
            if pairs:
                disp_thr = float(cfg.smd.min_disp_pair)
                new_pairs = []
                keepL_idx = set()
                keepR_idx = set()
                for (i, j) in pairs:
                    if i < len(boxesL) and j < len(boxesR):
                        disp = float(boxesL[i].cx - boxesR[j].cx)
                        if abs(disp) >= disp_thr:
                            new_pairs.append((i, j))
                            keepL_idx.add(i)
                            keepR_idx.add(j)

                if new_pairs:
                    idxL = sorted(keepL_idx)
                    idxR = sorted(keepR_idx)
                    mapL = {old: k for k, old in enumerate(idxL)}
                    mapR = {old: k for k, old in enumerate(idxR)}
                    boxesL = [boxesL[k] for k in idxL]
                    boxesR = [boxesR[k] for k in idxR]
                    pairs = [(mapL[i], mapR[j]) for (i, j) in new_pairs]
                else:
                    boxesL, boxesR, pairs = [], [], []

            # --- трекер
            keepL: Set[int] = trL.update(boxesL)
            keepR: Set[int] = trR.update(boxesR)

            # применяем фильтр к спискам боксов
            def _filter_boxes(boxes, keep: Set[int]):
                return [b for i, b in enumerate(boxes) if i in keep]

            boxesL = _filter_boxes(boxesL, keepL)
            boxesR = _filter_boxes(boxesR, keepR)

            # перестроить пары под отфильтрованные
            if pairs:
                # создадим индекс-переотображение
                idxL = {i: k for k, i in enumerate(sorted(keepL))}
                idxR = {j: k for k, j in enumerate(sorted(keepR))}
                new_pairs = []
                for (i, j) in pairs:
                    if i in idxL and j in idxR:
                        new_pairs.append((idxL[i], idxR[j]))
                pairs = new_pairs

            # визуализация
            visL = rL.copy()
            visR = rR.copy()
            draw_boxes(visL, boxesL, (0, 255, 0), "L")
            draw_boxes(visR, boxesR, (255, 0, 0), "R")
            vis = stack_lr(visL, visR)

            # запись/показ
            if writer is not None:
                writer.write(vis)

            if cfg.display:
                imshow_resized("VisualRadar L|R", vis, cfg.display_max_w, cfg.display_max_h)
                if (cv.waitKey(1) & 0xFF) == 27:
                    break

            # снапшоты
            if saver is not None and pairs:
                Q = getattr(calib, "Q", None) if hasattr(calib, "Q") else None
                for (i, j) in pairs:
                    if i >= len(boxesL) or j >= len(boxesR):
                        continue
                    bl = boxesL[i]
                    br = boxesR[j]
                    disp = float(bl.cx - br.cx)
                    saver.maybe_save(
                        rL,
                        rR,
                        (int(bl.x), int(bl.y), int(bl.w), int(bl.h)),
                        (int(br.x), int(br.y), int(br.w), int(br.h)),
                        disp,
                        Q=Q,
                    )

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
