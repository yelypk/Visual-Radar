import argparse, sys, os, logging, yaml, time
from dataclasses import asdict, replace, is_dataclass
from typing import Any, Mapping
from .config import AppConfig, SMDParams
from .pipeline import build_runtime, process_pair, render_and_record
from .utils import monotonic_s

log = logging.getLogger("visual_radar.cli")

def _merge_dc(dc, patch: Mapping[str, Any]):
    if not is_dataclass(dc): return dc
    d = asdict(dc)
    for k, v in patch.items():
        if k not in d: continue
        if is_dataclass(getattr(dc, k)):
            setattr(dc, k, _merge_dc(getattr(dc,k), v))
        else:
            setattr(dc, k, v)
    return dc

def build_parser():
    p = argparse.ArgumentParser("visual-radar")
    p.add_argument("--config", type=str, help="YAML with AppConfig/SMDParams fields")
    p.add_argument("--left", type=str); p.add_argument("--right", type=str)
    p.add_argument("--display", action="store_true"); p.add_argument("--print_fps", action="store_true")
    p.add_argument("--save_vis", action="store_true"); p.add_argument("--save_path", type=str, default="out.mp4")
    return p

def args_to_config(args)->AppConfig:
    cfg = AppConfig()  
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        cfg = _merge_dc(cfg, doc)
        if "smd" in doc: cfg.smd = _merge_dc(cfg.smd, doc["smd"])
    if args.left:  cfg.left  = args.left
    if args.right: cfg.right = args.right
    if args.display: cfg.display = True
    if args.print_fps: cfg.print_fps = True
    if args.save_vis:  cfg.save_vis  = True
    if args.save_path: cfg.save_path = args.save_path
    return cfg

def run(cfg: AppConfig):
    import time
    import cv2 as cv
    log = logging.getLogger("visual_radar.run")

    # build
    L, R, calib, detector, tracker, snapper, writer = build_runtime(cfg)
    try:
        detector.proj_mode = bool(getattr(calib, "proj_mode", False))
    except Exception:
        pass
    log.info("runtime ready: %dx%d proj_mode=%s", cfg.width, cfg.height, getattr(calib, "proj_mode", False))

    # FPS + heartbeat
    last_t = time.perf_counter(); fcount = 0
    debug_every = int(getattr(cfg, "debug_every_n", 30) or 30)
    want_debug = str(getattr(cfg, "log_level", "INFO")).upper() == "DEBUG"

    while True:
        t0 = time.perf_counter()

        okL, frameL, _ = L.read()
        okR, frameR, _ = R.read()
        if not (okL and okR):
            log.warning("read fail (L=%s R=%s) — reopen()", okL, okR)
            if not okL:
                try: L.reopen(); log.info("L.reopen() ok")
                except Exception: log.exception("L.reopen() failed")
            if not okR:
                try: R.reopen(); log.info("R.reopen() ok")
                except Exception: log.exception("R.reopen() failed")
            cv.waitKey(1)
            continue
        t1 = time.perf_counter()

        try:
            proc = process_pair(cfg, detector, tracker, calib, frameL, frameR)
        except Exception:
            log.exception("process_pair() failed")
            cv.waitKey(1)
            continue
        t2 = time.perf_counter()

        if snapper is not None:
            for k in getattr(proc, "show_idx", []):
                iL, iR = proc.pairs[k]
                disp = proc.boxesL[iL].cx() - proc.boxesR[iR].cx()
                snapper.maybe_save(proc.rectL, proc.rectR, proc.boxesL[iL], proc.boxesR[iR], disp, getattr(calib, "Q", None))

        fps_txt = None
        if cfg.print_fps:
            fcount += 1
            now = time.perf_counter()
            if now - last_t >= 1.0:
                fps_txt = f"FPS:{fcount/(now-last_t):.1f}"
                last_t = now; fcount = 0

        try:
            render_and_record(cfg, proc, tracker, writer, fps_txt=fps_txt)
        except Exception:
            log.exception("render_and_record() failed")
        t3 = time.perf_counter()

        # heartbeat раз в N кадров
        if want_debug and (fcount % debug_every == 0):
            log.info(
                "tick read=%.1fms proc=%.1fms draw=%.1fms  boxesL=%d boxesR=%d pairs=%d proj=%s",
                (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
                len(getattr(proc, "boxesL", [])),
                len(getattr(proc, "boxesR", [])),
                len(getattr(proc, "pairs", [])),
                getattr(calib, "proj_mode", False),
            )

        if cfg.display:
            cv.waitKey(1)


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = args_to_config(args)
    run(cfg)

if __name__ == "__main__":
    main()
