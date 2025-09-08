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

def _set_env(k: str, v: Any):
    # строковое значение для переменных окружения
    os.environ[str(k)] = str(v)

def _apply_env_overrides_from_yaml(doc: Mapping[str, Any] | None):
    """Позволяет настраивать ректификацию из YAML:
       варианты:
         env:
           VR_RECTIFY_ALPHA: 1.0
           VR_RECTIFY_PRESERVE_K: 1
           VR_RECTIFY_SCALE: 1.35
           VR_ROLL_DEG: 5
         rectify:
           alpha: 1.0
           preserve_k: true
           scale: 1.35
           roll_deg: 5
         fisheye:
           use: false
           balance: 0.4
    """
    if not isinstance(doc, dict):
        return

    # 1) прямой блок env:
    env_block = doc.get("env", {})
    if isinstance(env_block, dict):
        for k, v in env_block.items():
            _set_env(k, v)

    # 2) удобочитаемые блоки:
    rect = doc.get("rectify", {}) or {}
    if isinstance(rect, dict):
        if "alpha" in rect:       _set_env("VR_RECTIFY_ALPHA", rect["alpha"])
        if "preserve_k" in rect:  _set_env("VR_RECTIFY_PRESERVE_K", 1 if rect["preserve_k"] else 0)
        if "scale" in rect:       _set_env("VR_RECTIFY_SCALE", rect["scale"])
        if "roll_deg" in rect:    _set_env("VR_ROLL_DEG", rect["roll_deg"])

    fish = doc.get("fisheye", {}) or {}
    if isinstance(fish, dict):
        if "use" in fish:         _set_env("VR_USE_FISHEYE", 1 if fish["use"] else 0)
        if "balance" in fish:     _set_env("VR_FISHEYE_BALANCE", fish["balance"])

def _apply_env_overrides_from_args(args):
    # CLI-переключатели приоритетнее YAML
    if getattr(args, "rectify_alpha", None) is not None:
        _set_env("VR_RECTIFY_ALPHA", args.rectify_alpha)
    if getattr(args, "rectify_preserve_k", False):
        _set_env("VR_RECTIFY_PRESERVE_K", 1)
    if getattr(args, "rectify_scale", None) is not None:
        _set_env("VR_RECTIFY_SCALE", args.rectify_scale)
    if getattr(args, "fisheye", False):
        _set_env("VR_USE_FISHEYE", 1)
    if getattr(args, "fisheye_balance", None) is not None:
        _set_env("VR_FISHEYE_BALANCE", args.fisheye_balance)
    if getattr(args, "roll_deg", None) is not None:
        _set_env("VR_ROLL_DEG", args.roll_deg)

def build_parser():
    p = argparse.ArgumentParser("visual-radar")
    p.add_argument("--config", type=str, help="YAML with AppConfig/SMDParams fields")
    p.add_argument("--left", type=str); p.add_argument("--right", type=str)
    p.add_argument("--display", action="store_true"); p.add_argument("--print_fps", action="store_true")
    p.add_argument("--save_vis", action="store_true"); p.add_argument("--save_path", type=str, default="out.mp4")

    # --- новые удобные флаги для ректификации (альтернатива YAML) ---
    p.add_argument("--rectify-alpha", type=float, dest="rectify_alpha",
                   help="0..1: сколько сохранять FOV при stereoRectify (иначе берётся из YAML)")
    p.add_argument("--rectify-preserve-k", action="store_true", dest="rectify_preserve_k",
                   help="не менять масштаб/центр (используется в calibration.py)")
    p.add_argument("--rectify-scale", type=float, dest="rectify_scale",
                   help="доп. масштаб fx,fy при preserve_k (напр. 1.3)")
    p.add_argument("--fisheye", action="store_true",
                   help="включить fisheye-ректификацию (если поддержано калибровкой)")
    p.add_argument("--fisheye-balance", type=float, dest="fisheye_balance",
                   help="баланс кропа для fisheye (0..1)")

    # --- общий поворот обеих камер по часовой стрелке (градусы) ---
    p.add_argument("--roll-deg", type=float, dest="roll_deg",
                   help="Общий поворот обоих видоискателей по часовой стрелке (градусы) после rectification.")
    return p

def args_to_config(args)->AppConfig:
    cfg = AppConfig()
    doc = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        cfg = _merge_dc(cfg, doc)
        if "smd" in doc: cfg.smd = _merge_dc(cfg.smd, doc["smd"])

        # применяем YAML-овские оверрайды для ректификации
        _apply_env_overrides_from_yaml(doc)

    # CLI-аргументы поверх YAML
    if args.left:  cfg.left  = args.left
    if args.right: cfg.right = args.right
    if args.display: cfg.display = True
    if args.print_fps: cfg.print_fps = True
    if args.save_vis:  cfg.save_vis  = True
    if args.save_path: cfg.save_path = args.save_path

    _apply_env_overrides_from_args(args)
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

    # выведем, что именно включено для ректификации (полезно для отладки)
    log.info("rectify: alpha=%s preserve_k=%s scale=%s fisheye=%s balance=%s roll_deg=%s",
             os.getenv("VR_RECTIFY_ALPHA", ""),
             os.getenv("VR_RECTIFY_PRESERVE_K", ""),
             os.getenv("VR_RECTIFY_SCALE", ""),
             os.getenv("VR_USE_FISHEYE", ""),
             os.getenv("VR_FISHEYE_BALANCE", ""),
             os.getenv("VR_ROLL_DEG", ""))

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


