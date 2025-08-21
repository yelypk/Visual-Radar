
from typing import Tuple, List, Set
from .config import SMDParams
from .motion import as_gray, DualBGModel, find_motion_bboxes
from .stereo import gate_pairs_rectified, epipolar_ncc_match
from .utils import BBox

class StereoMotionDetector:
    def __init__(self, frame_size: Tuple[int,int], params: SMDParams):
        self.params = params
        w,h = frame_size
        self.bgL = DualBGModel((h,w))
        self.bgR = DualBGModel((h,w))

    def step(self, rectL_bgr, rectR_bgr):
        gL = as_gray(rectL_bgr)
        gR = as_gray(rectR_bgr)

        mL, boxesL = find_motion_bboxes(gL, self.bgL,
                                        self.params.min_area, self.params.max_area,
                                        self.params.thr_fast, self.params.thr_slow,
                                        use_clahe=self.params.use_clahe,
                                        size_aware_morph=self.params.size_aware_morph)
        mR, boxesR = find_motion_bboxes(gR, self.bgR,
                                        self.params.min_area, self.params.max_area,
                                        self.params.thr_fast, self.params.thr_slow,
                                        use_clahe=self.params.use_clahe,
                                        size_aware_morph=self.params.size_aware_morph)

        pairs = gate_pairs_rectified(boxesL, boxesR, self.params.y_eps, self.params.dmin, self.params.dmax)

        matched_R: Set[int] = set(j for _,j in pairs)
        for i,bl in enumerate(boxesL):
            if any(i==ii for ii,_ in pairs): 
                continue
            rb = epipolar_ncc_match(gL, gR, bl, self.params.stereo_search_pad, self.params.stereo_patch, self.params.stereo_ncc_min)
            if rb is not None:
                boxesR.append(rb)
                j = len(boxesR)-1
                pairs.append((i,j))
                matched_R.add(j)

        return mL, mR, boxesL, boxesR, pairs
