from dataclasses import dataclass
from typing import Optional, Tuple

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
    smd: SMDParams = SMDParams()
