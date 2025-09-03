from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SMDParams:
    night_auto: bool = False
    night_luma_thr: int = 60

    min_area: int = 12
    min_area_night_mult: float = 2.0
    morph_open_day: int = 3
    morph_open_night: int = 7
    noise_k_fast: float = 2.6
    noise_k_slow: float = 0.8

    persist_k: int = 5
    persist_m: int = 4

    crop_top: int = 0
    y_area_split: float = 0.65 

    stereo_patch: int = 13
    stereo_search_pad: int = 48
    stereo_ncc_min: float = 0.25

    stereo_only_sky: bool = False
    sails_only: bool = False
    sky_min_area_frac: float = 2e-5
    ground_min_area_frac: float = 4e-5

    min_disp_pair: float = 0.10


    min_track_speed: float = 0.0
    min_track_disp: float = 0.0

    enable_lr: bool = True
    lr_backtrack_pad: int = 8
    lr_patch: int = 13
    lr_ncc_min: float = 0.25
    lr_use_gradient: bool = True
    lr_grad_ksize: int = 3
    lr_subpixel: bool = True
    lr_max_cx_err_px: float = 2.0

@dataclass
class AppConfig:
    left: str = ''   
    right: str = ''   
    reader: str = "ffmpeg_mjpeg"  

    width: int = 1280
    height: int = 720

    calib_dir: str = "."                
    intrinsics: Optional[str] = None   
    baseline: Optional[float] = None   

    ffmpeg: str = "ffmpeg"
    mjpeg_q: int = 6
    ff_threads: int = 3
    cap_buffersize: int = 2
    read_timeout: float = 1.0  

    iou_thr: float = 0.3
    min_age: int = 3
    max_missed: int = 30
    min_disp_pair: float = 0.10  
    min_speed: float = 0.0

    snapshots: bool = False
    snap_dir: str = "snaps"
    snap_min_disp: float = 3.0
    snap_min_cc: float = 0.25
    snap_cooldown: float = 1.5
    snap_pad: int = 4
    snap_debug: bool = False

    display: bool = False
    display_max_w: int = 1920
    display_max_h: int = 1080
    print_fps: bool = False

    save_vis: bool = False
    save_path: str = "out.mp4"
    save_fps: float = 15.0


    sync_dt: float = 0.05
    sync_attempts: int = 4
    max_consec_fail: int = 30

    cv_threads: int = 0       
    use_optimized: bool = True
    drop_artifacts: bool = True

    smd: SMDParams = field(default_factory=SMDParams)
