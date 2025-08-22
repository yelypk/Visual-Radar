from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SMDParams:
    # --- Базовые стерео/геометрия ---
    y_eps: int = 16
    dmin: int = 1
    dmax: int = 400
    min_area: int = 25
    max_area: int = 0  # 0 = unlimited

    # Пороговые коэффициенты (обе формы для совместимости)
    thr_fast: Optional[float] = None
    thr_slow: Optional[float] = None
    noise_k_fast: float = 3.5   # если detector ждёт noise_k_*, они есть
    noise_k_slow: float = 2.0   # если ждёт thr_*, cli передаст их сюда

    # Порог движения (оптический поток/разница)
    min_flow: float = 0.25
    min_flow_small: float = 0.01

    # Стерео-NCC поиск
    stereo_patch: int = 13
    stereo_search_pad: int = 60
    stereo_ncc_min: float = 0.40

    # Анти-дрейф/сдвиг сцены
    drift_max_fg_pct: float = 0.25
    drift_max_frames: int = 60

    # Пре- и постобработка
    use_clahe: bool = True
    size_aware_morph: bool = True
    morph_open_day: int = 1
    morph_open_night: int = 3

    # День/ночь
    night_auto: bool = True
    night_luma_thr: float = 40.0
    min_area_night_mult: float = 4.0

    # Обрезка/маска
    crop_top: int = 0
    roi_mask: Optional[str] = None

    # --- Персистентность масок (НУЖНО ДЛЯ cli.py) ---
    persist_k: int = 3
    persist_m: int = 2

    # --- Параметры трекера (анти-мерцание + смещение) ---
    track_iou_thr: float = 0.3
    track_min_age: int = 3
    track_max_missed: int = 3
    track_min_disp: float = 6.0
    track_min_speed: float = 1.2

    # --- Parallax/area-гейты для ветра/дождя ---
    min_disp_pair: float = 2.5     # минимальный |диспаритет| по центрам L/R
    y_area_boost: float = 1.6      # множитель min_area для верхней/дальней зоны
    y_area_split: float = 0.55     # граница по высоте (0..1), выше — применяем boost


@dataclass
class AppConfig:
    left: str
    right: str

    calib_dir: str = "stereo_rtsp_out"
    intrinsics: Optional[str] = None
    baseline: Optional[float] = None

    width: Optional[int] = None
    height: Optional[int] = None
    native_resolution: bool = False

    display: bool = False
    headless: bool = False

    # Визуализация/запись
    save_vis: Optional[str] = None  # совместимо с ранними версиями
    save_path: str = "out.mp4"
    save_fps: float = 15.0

    smd: SMDParams = field(default_factory=SMDParams)

    # Источники / транскодирование
    reader: str = "opencv"      # "opencv" | "ffmpeg_mjpeg"
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
