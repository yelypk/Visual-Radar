from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple, Set, Optional

import cv2 as cv
import numpy as np

from .config import AppConfig, SMDParams
from .calibration import load_calibration, rectified_pair
from .io import open_stream, make_writer
from .visualize import draw_boxes, stack_lr, imshow_resized
from .snapshots import SnapshotSaver
from .tracks import BoxTracker
from .detector import StereoMotionDetector
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
    p.add_argument("--ff_threads", type=int, default=3, help="потоки декодера")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    # калибровка/ремап
    p.add_argument("--calib_dir", type=str, default="stereo_rtsp_out")
    p.add_argument("--intrinsics", type=str, default=None, help="путь к intrinsics.npz (опционально)")
    p.add_argument("--baseline", type=float, default=None, help="база камер в метрах (опционально)")

    # вывод
    p.add_argument("--display", action="store_true")
    p.add_argument("--display_max_w", type=int, default=1920)
    p.add_argument("--display_max_h", type=int, default=1080)
    p.add_argument("--save_vis", action="store_true")
    p.add_argument("--save_path", type=str, default="out.mp4")
    p.add_argument("--save_fps", type=float, default=15.0)

    # ночной режим
    p.add_argument("--night_auto", action="store_true")
    p.add_argument("--night_luma_thr", type=float, default=50.0)
    p.add_argument("--min_area_night_mult", type=float, default=4.0)
    p.add_argument("--morph_open_night", type=int, default=7)

    # день
    p.add_argument("--morph_open_day", type=int, default=3)
    p.add_argument("--min_area", type=int, default=18)
    p.add_argument("--max_area", type=int, default=0)

    # шум/порог движения
    p.add_argument("--noise_k_fast", type=float, default=2.0)
    p.add_argument("--noise_k_slow", type=float, default=1.0)
    p.add_argument("--min_flow", type=float, default=0.6)
    p.add_argument("--min_flow_small", type=float, default=0.2)

    # персистентность масок
    p.add_argument("--persist_k", type=int, default=3)
    p.add_argument("--persist_m", type=int, default=2)

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
    p.add_argument("--min_track_disp", type=float, default=6.0,
                   help="минимальное суммарное смещение трека (px) за окно")
    p.add_argument("--min_track_speed", type=float, default=1.2,
                   help="средняя скорость (px/кадр) за окно")

    # NEW: stereo parallax + Y-aware area
    p.add_argument("--min_disp_pair", type=float, default=2.5,
                   help="минимальная |диспаритет| по центрам боксов (анти-«дрожащий дальний фон»)")
    p.add_argument("--y_area_boost", type=float, default=1.6,
                   help="множитель min_area для верхней/дальней части кадра")
    p.add_argument("--y_area_split", type=float, default=0.55,
                   help="граница по высоте (0..1), выше которой применяется y_area_boost")

    # снапшоты
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--snap_dir", type=str, default="detections")
    p.add_argument("--snap_min_cc", type=float, default=0.6)
    p.add_argument("--snap_min_disp", type=float, default=1.5)
    p.add_argument("--snap_pad", type=int, default=4)
    p.add_argument("--snap_cooldown", type=float, default=1.5)
    p.add_argument("--snap_debug", action="store_true")

    return p


def args_to_config(args) -> AppConfig:
    smd = SMDParams(
        # день/ночь
        night_auto=args.night_auto,
        night_luma_thr=args.night_luma_thr,
        min_area=int(args.min_area),
        max_area=int(args.max_area),
        min_area_night_mult=float(args.min_area_night_mult),
        morph_open_day=int(args.morph_open_day),
        morph_open_night=int(args.morph_open_night),

        # движение/шум
        thr_fast=float(args.noise_k_fast),
        thr_slow=float(args.noise_k_slow),
        min_flow=float(args.min_flow),
        min_flow_small=float(args.min_flow_small),

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

        # NEW: stereo parallax + y-aware area
        min_disp_pair=float(args.min_disp_pair),
        y_area_boost=float(args.y_area_boost),
        y_area_split=float(args.y_area_split),
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

        display=args.display,
        headless=not args.display,
        save_vis=args.save_vis,

        smd=smd,

        # reader
        reader=args.reader,
        ffmpeg=args.ffmpeg,
        mjpeg_q=args.mjpeg_q,
        ff_threads=args.ff_threads,

        # вывод
        display_max_w=args.display_max_w,
        display_max_h=args.display_max_h,

        # снапшоты
        snapshots=args.snapshots,
        snap_dir=args.snap_dir,
        snap_min_cc=args.snap_min_cc,
        snap_min_disp=args.snap_min_disp,
        snap_pad=args.snap_pad,
        snap_cooldown=args.snap_cooldown,
        snap_debug=args.snap_debug,

        # запись
        save_path=args.save_path,
        save_fps=args.save_fps,
    )


# -----------------------------
# helpers
# -----------------------------
def warmup(reader, name: str, timeout_s: float = 8.0):
    """Дождаться первого кадра от источника (или дать шанс на fallback)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, fr, ts = reader.read()
        if ok:
            return ok, fr, ts
    return False, None, None


# -----------------------------
# main run loop
# -----------------------------
def run(cfg: AppConfig):
    # открываем источники
    L = open_stream(cfg.left, cfg.width, cfg.height,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)
    R = open_stream(cfg.right, cfg.width, cfg.height,
                    reader=cfg.reader, ffmpeg=cfg.ffmpeg,
                    mjpeg_q=cfg.mjpeg_q, ff_threads=cfg.ff_threads)

    # тёплый старт
    okL, fL0, _ = warmup(L, "LEFT", timeout_s=8.0)
    okR, fR0, _ = warmup(R, "RIGHT", timeout_s=8.0)
    if not okL or not okR:
        print("[warmup] no frames within 8s. stop.")
        try:
            L.release(); R.release()
        except Exception:
            pass
        return

    # калибровка (сигнатура: calib_dir, intrinsics_npz, frame_size, baseline_m)
    calib = load_calibration(
        Path(cfg.calib_dir),
        Path(cfg.intrinsics) if cfg.intrinsics else None,
        (cfg.width, cfg.height),
        cfg.baseline
    )

    # детектор и трекеры
    frame_size: Tuple[int, int] = (cfg.width, cfg.height)
    det = StereoMotionDetector(frame_size, cfg.smd)

    trL = BoxTracker(
        iou_thr=float(getattr(cfg.smd, "track_iou_thr", 0.3)),
        min_age=int(getattr(cfg.smd, "track_min_age", 3)),
        max_missed=int(getattr(cfg.smd, "track_max_missed", 3)),
        min_disp=float(getattr(cfg.smd, "track_min_disp", 6.0)),
        min_speed=float(getattr(cfg.smd, "track_min_speed", 1.2)),
    )
    trR = BoxTracker(
        iou_thr=float(getattr(cfg.smd, "track_iou_thr", 0.3)),
        min_age=int(getattr(cfg.smd, "track_min_age", 3)),
        max_missed=int(getattr(cfg.smd, "track_max_missed", 3)),
        min_disp=float(getattr(cfg.smd, "track_min_disp", 6.0)),
        min_speed=float(getattr(cfg.smd, "track_min_speed", 1.2)),
    )

    # снапшоты (опционально)
    saver: Optional[SnapshotSaver] = None
    if bool(getattr(cfg, "snapshots", False)):
        saver = SnapshotSaver(
            out_dir=getattr(cfg, "snap_dir", "detections"),
            min_disp=float(getattr(cfg, "snap_min_disp", 1.5)),
            min_cc=float(getattr(cfg, "snap_min_cc", 0.6)),
            cooldown=float(getattr(cfg, "snap_cooldown", 1.5)),
            pad=int(getattr(cfg, "snap_pad", 4)),
            debug=bool(getattr(cfg, "snap_debug", False)),
        )

    # запись (опционально)
    writer = None
    if bool(getattr(cfg, "save_vis", False)):
        sp = Path(getattr(cfg, "save_path", "out.mp4"))
        writer = make_writer(str(sp), (cfg.width * 2, cfg.height), fps=float(getattr(cfg, "save_fps", 15.0)))

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

            # --- Y-aware area gate (сильнее режем мелочь в верхней части)
            h = rL.shape[0]

            def _area_gate_y(boxes):
                out = []
                base_min_area = int(getattr(det.params, 'min_area', 0))
                boost = float(getattr(cfg.smd, 'y_area_boost', 1.6))
                split = float(getattr(cfg.smd, 'y_area_split', 0.55))
                thr_y = split * h
                for b in boxes:
                    minA = base_min_area
                    if b.cy < thr_y:  # верх/даль
                        minA = int(minA * boost)
                    if b.w * b.h >= minA:
                        out.append(b)
                return out

            boxesL = _area_gate_y(boxesL)
            boxesR = _area_gate_y(boxesR)

            # --- Stereo parallax gate (отбрасываем пары с почти нулевым диспаритетом)
            if pairs:
                disp_thr = float(getattr(cfg.smd, "min_disp_pair", 2.5))
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

            # --- гистерезис-трекер: оставляем устойчивые/движущиеся боксы
            keepL: Set[int] = trL.update(boxesL)
            keepR: Set[int] = trR.update(boxesR)

            def _filter_boxes_and_map(boxes, keep: Set[int]):
                remap = {}
                new_boxes = []
                for i, b in enumerate(boxes):
                    if i in keep:
                        remap[i] = len(new_boxes)
                        new_boxes.append(b)
                return new_boxes, remap

            boxesL, mapL = _filter_boxes_and_map(boxesL, keepL)
            boxesR, mapR = _filter_boxes_and_map(boxesR, keepR)

            # ✅ ВСЕГДА безопасно перестраиваем пары
            new_pairs = []
            for (i, j) in pairs or []:
                # если карты пустые — возвращаем исходный индекс, но проверяем границы
                ii = mapL.get(i, i)
                jj = mapR.get(j, j)
                if ii is not None and jj is not None and ii < len(boxesL) and jj < len(boxesR):
                    new_pairs.append((ii, jj))
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

            # снапшоты (с безопасной проверкой индексов)
            if saver is not None and pairs:
                Q = getattr(calib, "Q", None) if hasattr(calib, "Q") else None
                for (i, j) in pairs:
                    if i >= len(boxesL) or j >= len(boxesR):
                        continue
                    bl = boxesL[i]; br = boxesR[j]
                    disp = float(bl.cx - br.cx)
                    saver.maybe_save(
                        rL, rR,
                        (int(bl.x), int(bl.y), int(bl.w), int(bl.h)),
                        (int(br.x), int(br.y), int(br.w), int(br.h)),
                        disp, Q=Q
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
            L.release(); R.release()
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
