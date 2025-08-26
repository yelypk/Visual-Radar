from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SMDParams:
    night_auto: bool = True
    night_luma_thr: float = 50.0

    min_area: int = 25
    max_area: int = 0
    min_area_night_mult: float = 4.0

    morph_open_day: int = 3
    morph_open_night: int = 7
    thr_fast: float = 2.0
    thr_slow: float = 1.0
    min_flow: float = 0.6
    min_flow_small: float = 0.2
    size_aware_morph: bool = True

    crop_top: int = 0
    roi_mask: Optional[str] = None


    persist_k: int = 4
    persist_m: int = 3
    drift_max_fg_pct: float = 0.25
    drift_max_frames: int = 60

    track_iou_thr: float = 0.3
    track_min_age: int = 3
    track_max_missed: int = 3
    track_min_disp: float = 4.0      # было 6.0
    track_min_speed: float = 0.6     # было 0.8

    y_eps: float = 6.0
    dmin: float = -512.0
    dmax: float = 512.0
    stereo_search_pad: int = 64
    stereo_patch: int = 13
    stereo_ncc_min: float = 0.25
    min_disp_pair: float = 2.5

    y_area_boost: float = 1.6
    y_area_split: float = 0.62       

    use_clahe: bool = True
    sails_only: bool = False
    sails_water_split: float = 0.55
    sails_white_delta: float = 18.0
    sails_min_h_over_w: float = 0.85
    sails_max_w: int = 240
    sails_min_h: int = 6

@dataclass
class AppConfig:
    left: str = ""
    right: str = ""
    calib_dir: str = "stereo_rtsp_out"
    intrinsics: Optional[str] = None
    baseline: Optional[float] = None
    width: int = 1280
    height: int = 720

    display: bool = False
    save_vis: bool = False

    smd: SMDParams = field(default_factory=SMDParams)

    reader: str = "opencv"           
    ffmpeg: str = "ffmpeg"
    mjpeg_q: int = 6
    ff_threads: int = 3

    display_max_w: int = 1920
    display_max_h: int = 1080
    window: str = "normal"

    snapshots: bool = False
    snap_dir: str = "detections"
    snap_min_cc: float = 0.6
    snap_min_disp: float = 1.5
    snap_pad: int = 4
    snap_cooldown: float = 1.5
    snap_debug: bool = False


    save_path: str = "out.mp4"
    save_fps: float = 15.0

    sync_max_dt: float = 0.05
    sync_attempts: int = 4

    cap_buffersize: int = 1
    stall_timeout: float = 3.0
    max_consec_fail: int = 30

    cv_threads: int = 0
    use_optimized: bool = True
    print_fps: bool = False
    drop_artifacts: bool = True
    