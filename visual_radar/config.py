
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SMDParams:
    y_eps:int = 16
    dmin:int = 1
    dmax:int = 400
    min_area:int = 25
    max_area:int = 0  # 0 = unlimited
    thr_fast: Optional[float] = None
    thr_slow: Optional[float] = None
    min_flow: float = 0.25
    min_flow_small: float = 0.01
    stereo_patch:int = 13
    stereo_search_pad:int = 60
    stereo_ncc_min:float = 0.40
    use_clahe: bool = True
    size_aware_morph: bool = True
    # --- Night & noise-aware ---
    night_auto: bool = True              # авто-переключение по яркости
    night_luma_thr: float = 40.0         # порог Y (0..255), ниже = ночь
    min_area_night_mult: float = 4.0     # во сколько раз увеличить min_area ночью
    noise_k_fast: float = 3.5            # множитель σ для быстрого диффа
    noise_k_slow: float = 2.0            # множитель σ для медленного диффа
    morph_open_day: int = 1              # размер ядра открытия днём
    morph_open_night: int = 3            # размер ядра открытия ночью
    crop_top: int = 0                    # срез сверху в пикселях (обрезать небо)
    roi_mask: Optional[str] = None       # PNG-маска (белое=используем, чёрное=глушим)

@dataclass
class AppConfig:
    left:str
    right:str
    calib_dir:str = "stereo_rtsp_out"
    intrinsics: Optional[str] = None
    baseline: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    native_resolution: bool = False
    display: bool = False
    headless: bool = False
    save_vis: Optional[str] = None
    smd: SMDParams = field(default_factory=SMDParams)

    # Stream / transcoding
    reader: str = "opencv"            # "opencv" | "ffmpeg_mjpeg"
    ffmpeg: str = "ffmpeg"
    mjpeg_q: int = 6
    ff_threads: int = 3

    # Display
    display_max_w: int = 1280
    display_max_h: int = 720

    # Snapshots
    snapshots: bool = True
    snap_dir: str = "detections"
    snap_min_disp: float = 1.5
    snap_min_cc: float = 0.60
    snap_min_z: float = 0.0
    snap_max_z: float = 5000.0
    snap_cooldown: float = 1.5
    snap_pad: int = 4
    snap_debug: bool = False
