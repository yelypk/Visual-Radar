from __future__ import annotations

from collections import deque
from typing import List, Set, Tuple, Optional
import os

import numpy as np
import cv2 as cv

from visual_radar.config import SMDParams
from visual_radar.motion import (
    as_gray,
    DualBGModel,
    make_masks_static_and_slow,
)
from visual_radar.stereo import (
    gate_pairs_rectified,
    epipolar_ncc_match,
    refine_pairs_lr_consistency,
)
from visual_radar.utils import BBox


# -------------------------- small utils --------------------------

def _vr_pre(gray: np.ndarray, night_auto: bool) -> np.ndarray:
    if night_auto:
        try:
            gray = cv.bilateralFilter(gray, 5, 12, 12)
        except Exception:
            pass
        gray = cv.GaussianBlur(gray, (5, 5), 0)
    return gray


def _boxes_from_mask(mask: np.ndarray, min_area: int, max_area: int) -> List[BBox]:
    boxes: List[BBox] = []
    if mask is None or mask.size == 0:
        return boxes
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        area = int(w) * int(h)
        if area < max(min_area, 1):
            continue
        if max_area and area > max_area:
            continue
        boxes.append(BBox(float(x), float(y), float(w), float(h)))
    return boxes


def _shape_gate_sky(boxes: List[BBox], gray: np.ndarray, y_split: int) -> List[BBox]:
    if not boxes:
        return boxes
    H, _ = gray.shape[:2]
    y_split = int(np.clip(y_split, 0, H - 1))
    out: List[BBox] = []
    for b in boxes:
        # длинные узкие полосы вверху (облака/лучи)
        if b.y < y_split and b.w > 4 * b.h and b.h < 12:
            continue
        out.append(b)
    return out


def _vehicles_only_gate(
    boxes: List[BBox],
    gray: np.ndarray,
    y_min_frac: float,
    min_ar: float,
    max_ar: float,
    min_w: int,
    max_h: int,
) -> List[BBox]:
    if not boxes:
        return boxes
    H, W = gray.shape[:2]
    y_min = int(np.clip(H * y_min_frac, 0, H - 1))
    keep: List[BBox] = []
    for b in boxes:
        if b.y < y_min:
            continue
        ar = float(b.w) / max(1.0, float(b.h))
        if ar < min_ar or ar > max_ar:
            continue
        if b.w < min_w:
            continue
        if max_h > 0 and b.h > max_h:
            continue
        keep.append(b)
    return keep


def _glare_filter_boxes(
    boxes: List[BBox],
    frame_bgr: np.ndarray,
    s_max: int,
    v_min: int,
    pillar_max_w: int,
    pillar_ar_min: float,
) -> List[BBox]:
    """Антиблик: узкие «столбы» и почти белые области (низкая S, высокая V)."""
    if not boxes:
        return boxes
    H, W = frame_bgr.shape[:2]
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    keep: List[BBox] = []
    use_pillar = pillar_max_w > 0 and pillar_ar_min > 0.0
    use_hsv = (s_max > 0) or (v_min > 0)

    for b in boxes:
        w, h = int(b.w), int(b.h)
        if use_pillar and (w <= pillar_max_w) and (h / max(1, w) >= pillar_ar_min):
            continue
        if use_hsv:
            x1, y1 = max(0, int(b.x)), max(0, int(b.y))
            x2, y2 = min(W, int(b.x + b.w)), min(H, int(b.y + b.h))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = hsv[y1:y2, x1:x2]
            _, meanS, meanV, _ = cv.mean(roi)
            if (s_max > 0 and meanS <= s_max) and (v_min > 0 and meanV >= v_min):
                continue
        keep.append(b)
    return keep


def _edge_coherence_gate(
    boxes: List[BBox],
    gray: np.ndarray,
    *,
    coh_min: float,
    max_tilt_deg: float,
    min_w: int,
    min_h: int,
) -> List[BBox]:
    """Подавление водной «ряби» по структурному тензору."""
    if not boxes:
        return boxes
    H, W = gray.shape[:2]
    keep: List[BBox] = []

    for b in boxes:
        if b.w < min_w or b.h < min_h:
            continue
        x1, y1 = max(0, int(b.x)), max(0, int(b.y))
        x2, y2 = min(W, int(b.x + b.w)), min(H, int(b.y + b.h))
        if x2 <= x1 or y2 <= y1:
            continue

        patch = gray[y1:y2, x1:x2]
        gx = cv.Sobel(patch, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(patch, cv.CV_32F, 0, 1, ksize=3)

        Sxx = float(np.mean(gx * gx))
        Syy = float(np.mean(gy * gy))
        Sxy = float(np.mean(gx * gy))
        tr = Sxx + Syy
        if tr <= 1e-6:
            continue
        tmp = np.sqrt(max(0.0, (Sxx - Syy) * (Sxx - Syy) + 4.0 * Sxy * Sxy))
        lam1 = 0.5 * (tr + tmp)
        lam2 = 0.5 * (tr - tmp)

        coh = (lam1 - lam2) / (lam1 + lam2 + 1e-6)
        if coh < coh_min:
            continue

        theta = 0.5 * np.arctan2(2.0 * Sxy, (Sxx - Syy + 1e-12))
        tilt_deg = abs(np.degrees(theta))  # 0° ~ горизонт
        if tilt_deg > max_tilt_deg:
            continue

        keep.append(b)
    return keep


# -------------------------- LK optical flow validator --------------------------

def _lk_filter_boxes(
    gray: np.ndarray,
    prev_gray: Optional[np.ndarray],
    boxes: List[BBox],
    *,
    enable: bool = True,
    max_corners: int = 60,
    quality: float = 0.01,
    min_dist: int = 5,
    win_size: int = 21,
    max_level: int = 3,
    fb_thresh: float = 1.5,
    min_med_speed: float = 0.25,
    min_good_frac: float = 0.5,
) -> List[BBox]:
    """
    Оставляет только те боксы, внутри которых точки LK дают согласованное смещение.
    Отсекание «ложняков» от облаков/бликов при малом/хаотичном опт.потоке.
    """
    if not enable or prev_gray is None or not boxes:
        return boxes

    H, W = gray.shape[:2]
    keep: List[BBox] = []

    lk_params = dict(winSize=(int(win_size), int(win_size)),
                     maxLevel=int(max_level),
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    for b in boxes:
        x1, y1 = max(0, int(b.x)), max(0, int(b.y))
        x2, y2 = min(W, int(b.x + b.w)), min(H, int(b.y + b.h))
        if x2 <= x1 or y2 <= y1:
            continue

        # выбираем фичи внутри бокса
        roi = prev_gray[y1:y2, x1:x2]
        p0 = cv.goodFeaturesToTrack(roi, maxCorners=int(max_corners), qualityLevel=float(quality), minDistance=int(min_dist))
        if p0 is None or len(p0) < 4:
            # мало фич — не режем по LK (чтобы не потерять маленькие объекты)
            keep.append(b)
            continue

        # смещаем точки в глобальные координаты
        p0 = p0.reshape(-1, 1, 2).astype(np.float32)
        p0[:, 0, 0] += x1
        p0[:, 0, 1] += y1

        p1, st1, err1 = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        if p1 is None:
            keep.append(b);  # не наказываем
            continue
        # обратный прогон (forward-backward)
        p0b, st2, err2 = cv.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_params)
        good = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)

        if not np.any(good):
            keep.append(b)  # пусто — оставим, пусть решит следующая стадия
            continue

        p0g = p0[good].reshape(-1, 2)
        p1g = p1[good].reshape(-1, 2)
        p0bg = p0b[good].reshape(-1, 2)

        fb = np.linalg.norm(p0g - p0bg, axis=1)
        disp = np.linalg.norm(p1g - p0g, axis=1)

        ok = fb < float(fb_thresh)
        if not np.any(ok):
            continue

        disp_ok = disp[ok]
        good_frac = float(np.count_nonzero(ok)) / float(len(good))
        med_speed = float(np.median(disp_ok)) if disp_ok.size else 0.0

        if good_frac >= float(min_good_frac) and med_speed >= float(min_med_speed):
            keep.append(b)
        # иначе — отбрасываем бокс
    return keep


# -------------------------- ROI / masks --------------------------

def _norm_or_px_to_xy(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """Преобразует список точек (0..1 или px) в пиксели."""
    pts = points.astype(np.float32).copy()
    if np.all((pts >= 0.0) & (pts <= 1.0)):
        pts[:, 0] = pts[:, 0] * w
        pts[:, 1] = pts[:, 1] * h
    return pts.astype(np.int32)


def _parse_poly_str(s: str) -> List[np.ndarray]:
    """
    Разбор строки полигона:
      "x1,y1 x2,y2 x3,y3; x1,y1 x2,y2 ... " — несколько полигонов через ';'
      Значения могут быть нормированные (0..1) или пиксели.
    """
    polys: List[np.ndarray] = []
    if not s:
        return polys
    for poly_str in [p.strip() for p in s.split(";") if p.strip()]:
        pts = []
        for p in poly_str.replace("|", " ").split():
            if "," not in p:
                continue
            x, y = p.split(",")
            try:
                pts.append([float(x), float(y)])
            except Exception:
                pass
        if len(pts) >= 3:
            polys.append(np.array(pts, dtype=np.float32))
    return polys


def _make_roi_mask(w: int, h: int, keep_rect: Optional[Tuple[float, float, float, float]],
                   keep_poly: List[np.ndarray],
                   ignore_rects: List[Tuple[float, float, float, float]],
                   ignore_polys: List[np.ndarray]) -> np.ndarray:
    """Строит маску «что анализируем» (1) / «что игнорируем» (0)."""
    mask = np.ones((h, w), np.uint8)

    # keep rect (если задан) — всё остальное обнуляем
    if keep_rect:
        x0, y0, x1, y1 = keep_rect
        if 0 <= x0 <= 1 and 0 <= x1 <= 1 and 0 <= y0 <= 1 and 0 <= y1 <= 1:
            X0, Y0 = int(x0 * w), int(y0 * h)
            X1, Y1 = int(x1 * w), int(y1 * h)
        else:
            X0, Y0, X1, Y1 = int(x0), int(y0), int(x1), int(y1)
        X0 = np.clip(X0, 0, w - 1); X1 = np.clip(X1, X0 + 1, w)
        Y0 = np.clip(Y0, 0, h - 1); Y1 = np.clip(Y1, Y0 + 1, h)
        rect_mask = np.zeros_like(mask)
        rect_mask[Y0:Y1, X0:X1] = 1
        mask = cv.bitwise_and(mask, rect_mask)

    # keep polys — пересечение со всеми полигонами
    for poly in keep_poly:
        poly_px = _norm_or_px_to_xy(poly, w, h)
        pm = np.zeros_like(mask)
        cv.fillPoly(pm, [poly_px], 1)
        mask = cv.bitwise_and(mask, pm)

    # ignore rects
    for r in ignore_rects:
        x0, y0, x1, y1 = r
        if 0 <= x0 <= 1 and 0 <= x1 <= 1 and 0 <= y0 <= 1 and 0 <= y1 <= 1:
            X0, Y0 = int(x0 * w), int(y0 * h)
            X1, Y1 = int(x1 * w), int(y1 * h)
        else:
            X0, Y0, X1, Y1 = int(x0), int(y0), int(x1), int(y1)
        X0 = np.clip(X0, 0, w - 1); X1 = np.clip(X1, X0 + 1, w)
        Y0 = np.clip(Y0, 0, h - 1); Y1 = np.clip(Y1, Y0 + 1, h)
        mask[Y0:Y1, X0:X1] = 0

    # ignore polys
    for poly in ignore_polys:
        poly_px = _norm_or_px_to_xy(poly, w, h)
        cv.fillPoly(mask, [poly_px], 0)

    return mask


# -------------------------- detector --------------------------

class StereoMotionDetector:
    """Детектор движения + L/R-пары, с пер-камерными ROI-масками."""

    def __init__(self, frame_size: Tuple[int, int], params: SMDParams) -> None:
        self.w, self.h = int(frame_size[0]), int(frame_size[1])
        self.params = params

        algo = str(getattr(params, "bg_algo", "MOG2"))
        ds_bg = float(getattr(params, "bg_downscale", 1.0))
        hist_fast = int(getattr(params, "bg_history_fast", 200))
        hist_slow = int(getattr(params, "bg_history_slow", 1200))
        var_thr = float(getattr(params, "bg_var_threshold", 16.0))
        detect_shadows = bool(getattr(params, "bg_detect_shadows", False))
        lr = float(getattr(params, "bg_learning_rate", -1.0))

        self.bgL = DualBGModel(
            (self.h, self.w), algo=algo,
            history_fast=hist_fast, history_slow=hist_slow,
            var_threshold=var_thr, detect_shadows=detect_shadows,
            learning_rate=lr, downscale=ds_bg,
        )
        self.bgR = DualBGModel(
            (self.h, self.w), algo=algo,
            history_fast=hist_fast, history_slow=hist_slow,
            var_threshold=var_thr, detect_shadows=detect_shadows,
            learning_rate=lr, downscale=ds_bg,
        )

        self.persist_k = int(getattr(self.params, "persist_k", 5))
        self.need_hits = int(getattr(self.params, "persist_m", 3))
        self.histL: deque[np.ndarray] = deque(maxlen=self.persist_k)
        self.histR: deque[np.ndarray] = deque(maxlen=self.persist_k)
        self.sumL = np.zeros((self.h, self.w), np.uint16)
        self.sumR = np.zeros((self.h, self.w), np.uint16)

        self.proj_mode = False

        # glare/roi params + маски отдельно для L/R
        self._glr_inited = False
        self._keep_mask_L: Optional[np.ndarray] = None
        self._keep_mask_R: Optional[np.ndarray] = None
        self._glr: dict = {}

        # кадр «t-1» для LK-валидатора
        self._prev_grayL: Optional[np.ndarray] = None
        self._prev_grayR: Optional[np.ndarray] = None

        # управление LK через env/params
        self._lk_enable = int(os.getenv("VR_LK_ENABLE", "1") or "1") == 1
        self._lk_fb = float(os.getenv("VR_LK_FB_THR", "1.5") or "1.5")
        self._lk_min_speed = float(os.getenv("VR_LK_MIN_SPEED", "0.25") or "0.25")
        self._lk_good_frac = float(os.getenv("VR_LK_GOOD_FRAC", "0.5") or "0.5")
        self._lk_max_corners = int(os.getenv("VR_LK_MAX_CORNERS", "60") or "60")
        self._lk_quality = float(os.getenv("VR_LK_QUALITY", "0.01") or "0.01")
        self._lk_min_dist = int(os.getenv("VR_LK_MIN_DIST", "5") or "5")
        self._lk_win = int(os.getenv("VR_LK_WINSZ", "21") or "21")
        self._lk_levels = int(os.getenv("VR_LK_LEVELS", "3") or "3")

    def _init_glare_block(self, frame_shape: Tuple[int, int, int]) -> None:
        if self._glr_inited:
            return
        h, w = frame_shape[:2]

        # читаем общие и пер-камерные переменные
        def _env(k: str, default: str = "") -> str:
            return os.getenv(k, default)

        self._glr = {
            # антиблик
            "s_max": int(_env("VR_VETO_SMAX", "0") or "0"),
            "v_min": int(_env("VR_VETO_VMIN", "0") or "0"),
            "pillar_max_w": int(_env("VR_PILLAR_MAX_W", "0") or "0"),
            "pillar_ar_min": float(_env("VR_PILLAR_AR_MIN", "0") or "0"),
            # воды/кохерентность
            "wf_enable": int(_env("VR_WF_ENABLE", "1") or "1"),
            "wf_coh_min": float(_env("VR_WF_COH_MIN", "0.35") or "0.35"),
            "wf_max_tilt_deg": float(_env("VR_WF_MAX_TILT_DEG", "25") or "25"),
            "wf_min_w": int(_env("VR_WF_MIN_W", "12") or "12"),
            "wf_min_h": int(_env("VR_WF_MIN_H", "10") or "10"),
            # моно-режим принудительно
            "force_mono": int(_env("VR_FORCE_MONO", "0") or "0"),
        }

        # прямоугольные ROI (общие и L/R)
        keep_rect_g = _env("VR_KEEP_RECT", "")
        keep_rect_L = _env("VR_KEEP_RECT_L", keep_rect_g)
        keep_rect_R = _env("VR_KEEP_RECT_R", keep_rect_g)

        def _parse_rect(s: str) -> Optional[Tuple[float, float, float, float]]:
            if not s:
                return None
            try:
                x0, y0, x1, y1 = [float(v.strip()) for v in s.split(",")]
                return (x0, y0, x1, y1)
            except Exception:
                return None

        krL = _parse_rect(keep_rect_L)
        krR = _parse_rect(keep_rect_R)

        # полигоны keep / ignore (общие и L/R)
        kp_g = _parse_poly_str(_env("VR_ROI_POLY", ""))
        kp_L = _parse_poly_str(_env("VR_ROI_POLY_L", ""))
        kp_R = _parse_poly_str(_env("VR_ROI_POLY_R", ""))
        if kp_g and not kp_L:
            kp_L = kp_g
        if kp_g and not kp_R:
            kp_R = kp_g

        ip_g = _parse_poly_str(_env("VR_IGNORE_POLY", ""))
        ip_L = _parse_poly_str(_env("VR_IGNORE_POLY_L", ""))
        ip_R = _parse_poly_str(_env("VR_IGNORE_POLY_R", ""))
        if ip_g and not ip_L:
            ip_L = ip_g
        if ip_g and not ip_R:
            ip_R = ip_g

        # прямоугольники ignore
        ir_g = _env("VR_IGNORE_RECTS", "")
        def _parse_rects(s: str) -> List[Tuple[float, float, float, float]]:
            out = []
            for part in [p.strip() for p in s.split(";") if p.strip()]:
                try:
                    x0, y0, x1, y1 = [float(v.strip()) for v in part.split(",")]
                    out.append((x0, y0, x1, y1))
                except Exception:
                    pass
            return out

        ir_L = _parse_rects(_env("VR_IGNORE_RECTS_L", ir_g))
        ir_R = _parse_rects(_env("VR_IGNORE_RECTS_R", ir_g))

        # строим две независимые маски
        self._keep_mask_L = _make_roi_mask(w, h, krL, kp_L, ir_L, ip_L)
        self._keep_mask_R = _make_roi_mask(w, h, krR, kp_R, ir_R, ip_R)

        self._glr_inited = True

    def set_proj_mode(self, enabled: bool) -> None:
        self.proj_mode = bool(enabled)

    def _persist_update(self, sum_img: np.ndarray, q: deque, cur: np.ndarray) -> np.ndarray:
        if len(q) == self.persist_k:
            oldest = q.popleft()
            if oldest is not None:
                sum_img -= oldest.astype(np.uint16)
        q.append(cur)
        sum_img += cur.astype(np.uint16)
        thr = 255 * int(self.need_hits)
        return (sum_img >= thr).astype(np.uint8) * 255

    # -------------------------- main step --------------------------

    def step(
        self,
        rectL_bgr: np.ndarray,
        rectR_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[BBox], List[BBox], List[Tuple[int, int]]]:

        night_auto = bool(getattr(self.params, "night_auto", False))

        grayL = _vr_pre(as_gray(rectL_bgr), night_auto)
        grayR = _vr_pre(as_gray(rectR_bgr), night_auto)

        use_clahe = not night_auto
        thr_fast = float(getattr(self.params, "noise_k_fast", 2.4))
        thr_slow = float(getattr(self.params, "noise_k_slow", 0.9))
        morph_k = int(getattr(self.params, "morph_open_day", 3))
        if night_auto:
            morph_k = int(getattr(self.params, "morph_open_night", 7))

        mL_base, mL_slow = make_masks_static_and_slow(grayL, self.bgL, thr_fast, thr_slow, use_clahe=use_clahe, kernel_size=morph_k, night_auto=night_auto)
        mR_base, mR_slow = make_masks_static_and_slow(grayR, self.bgR, thr_fast, thr_slow, use_clahe=use_clahe, kernel_size=morph_k, night_auto=night_auto)

        y_split_frac = float(getattr(self.params, "y_area_split", 0.6))
        y_split = int(np.clip(int(self.h * y_split_frac), 0, self.h - 1))
        mL = mL_slow.copy(); mL[:y_split, :] = mL_base[:y_split, :]
        mR = mR_slow.copy(); mR[:y_split, :] = mR_base[:y_split, :]

        min_area = int(getattr(self.params, "min_area", 12))
        if night_auto:
            mult = float(getattr(self.params, "min_area_night_mult", 2.0))
            min_area = int(min_area * mult)
        max_area = int(getattr(self.params, "max_area", 0))
        ds_cc = float(getattr(self.params, "cc_downscale", 0.5))
        conn = int(getattr(self.params, "cc_connectivity", 4))
        max_boxes_cc = int(getattr(self.params, "stereo_max_boxes", 150))

        def _cc_clean(mm: np.ndarray) -> np.ndarray:
            H, W = mm.shape[:2]
            nz = int(np.count_nonzero(mm))
            if nz < 8 or nz > int(0.45 * H * W):
                return np.zeros((H, W), np.uint8)

            if ds_cc != 1.0:
                sm = cv.resize(mm, dsize=None, fx=ds_cc, fy=ds_cc, interpolation=cv.INTER_NEAREST)
                scale = 1.0 / ds_cc
                min_a_s = max(1, int(min_area * ds_cc * ds_cc))
                max_a_s = max(1, int(max_area * ds_cc * ds_cc)) if max_area else 0
            else:
                sm = mm
                scale = 1.0
                min_a_s = max(1, int(min_area))
                max_a_s = int(max_area)

            num, labels, stats, _ = cv.connectedComponentsWithStats(sm, connectivity=conn, ltype=cv.CV_32S)

            boxes = []
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if area < min_a_s:
                    continue
                if max_a_s and area > max_a_s:
                    continue
                X = int(round(x * scale));  Y = int(round(y * scale))
                Wb = int(round(w * scale)); Hb = int(round(h * scale))
                boxes.append((X, Y, Wb, Hb))

            if len(boxes) > max_boxes_cc:
                boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
                boxes = boxes[:max_boxes_cc]

            clean = np.zeros((H, W), np.uint8)
            for (x, y, w, h) in boxes:
                x2 = min(W, x + max(w, 1)); y2 = min(H, y + max(h, 1))
                if x < 0 or y < 0 or x >= x2 or y >= y2:
                    continue
                clean[y:y2, x:x2] = 255
            return clean

        mL_clean = _cc_clean(mL)
        mR_clean = _cc_clean(mR)

        mL_final = self._persist_update(self.sumL, self.histL, mL_clean)
        mR_final = self._persist_update(self.sumR, self.histR, mR_clean)

        # инициализация масок и применение отдельно для L/R
        self._init_glare_block(rectL_bgr.shape)
        if self._keep_mask_L is not None:
            mL_final = cv.bitwise_and(mL_final, self._keep_mask_L * 255)
        if self._keep_mask_R is not None:
            mR_final = cv.bitwise_and(mR_final, self._keep_mask_R * 255)

        boxesL = _boxes_from_mask(mL_final, min_area=min_area, max_area=max_area)
        boxesR = _boxes_from_mask(mR_final, min_area=min_area, max_area=max_area)

        boxesL = _shape_gate_sky(boxesL, grayL, y_split)
        boxesR = _shape_gate_sky(boxesR, grayR, y_split)

        # LK-валидатор (подавляет облака/блики)
        if self._lk_enable:
            boxesL = _lk_filter_boxes(
                grayL, self._prev_grayL, boxesL,
                enable=True,
                max_corners=self._lk_max_corners,
                quality=self._lk_quality,
                min_dist=self._lk_min_dist,
                win_size=self._lk_win,
                max_level=self._lk_levels,
                fb_thresh=self._lk_fb,
                min_med_speed=self._lk_min_speed,
                min_good_frac=self._lk_good_frac,
            )
            boxesR = _lk_filter_boxes(
                grayR, self._prev_grayR, boxesR,
                enable=True,
                max_corners=self._lk_max_corners,
                quality=self._lk_quality,
                min_dist=self._lk_min_dist,
                win_size=self._lk_win,
                max_level=self._lk_levels,
                fb_thresh=self._lk_fb,
                min_med_speed=self._lk_min_speed,
                min_good_frac=self._lk_good_frac,
            )

        # подавление «воды» (кохерентность)
        if int(self._glr.get("wf_enable", 1)) == 1:
            boxesL = _edge_coherence_gate(
                boxesL, grayL,
                coh_min=float(self._glr["wf_coh_min"]),
                max_tilt_deg=float(self._glr["wf_max_tilt_deg"]),
                min_w=int(self._glr["wf_min_w"]),
                min_h=int(self._glr["wf_min_h"]),
            )
            boxesR = _edge_coherence_gate(
                boxesR, grayR,
                coh_min=float(self._glr["wf_coh_min"]),
                max_tilt_deg=float(self._glr["wf_max_tilt_deg"]),
                min_w=int(self._glr["wf_min_w"]),
                min_h=int(self._glr["wf_min_h"]),
            )

        # антиблик
        s_max = int(self._glr.get("s_max", 0))
        v_min = int(self._glr.get("v_min", 0))
        pillar_w = int(self._glr.get("pillar_max_w", 0))
        pillar_ar = float(self._glr.get("pillar_ar_min", 0.0))
        if (s_max > 0 or v_min > 0 or pillar_w > 0):
            boxesL = _glare_filter_boxes(boxesL, rectL_bgr, s_max, v_min, pillar_w, pillar_ar)
            boxesR = _glare_filter_boxes(boxesR, rectR_bgr, s_max, v_min, pillar_w, pillar_ar)

        # обновим prev-кадры перед возможным ранним выходом (mono/proj)
        self._prev_grayL = grayL.copy()
        self._prev_grayR = grayR.copy()

        # если нужно — работаем строго в моно (в этих зонах часто нет L↔R совпадений)
        if self.proj_mode or int(self._glr.get("force_mono", 0)) == 1:
            return mL_final, mR_final, boxesL, boxesR, []

        # иначе — строим пары (gate → NCC-домэтч → LR-consistency refine)
        pairs = gate_pairs_rectified(
            boxesL, boxesR,
            float(getattr(self.params, "y_eps", 6.0)),
            float(getattr(self.params, "dmin", -512.0)),
            float(getattr(self.params, "dmax", 512.0)),
        )

        matched_R: Set[int] = set(j for _, j in pairs)
        for i, bl in enumerate(boxesL):
            if any(i == ii for ii, _ in pairs):
                continue
            rb = epipolar_ncc_match(
                grayL, grayR, bl,
                int(getattr(self.params, "stereo_search_pad", 96)),
                int(getattr(self.params, "stereo_patch", 9)),
                float(getattr(self.params, "stereo_ncc_min", 0.20)),
            )
            if rb is not None:
                boxesR.append(rb)
                j = len(boxesR) - 1
                pairs.append((i, j))
                matched_R.add(j)

        # финальное уточнение пар по L↔R согласованности
        pairs = refine_pairs_lr_consistency(
            boxesL, boxesR, pairs, grayL, grayR,
            backtrack_pad=int(getattr(self.params, "backtrack_pad", 8)),
            patch_size=int(getattr(self.params, "stereo_patch", 9)),
            ncc_min=float(getattr(self.params, "stereo_ncc_min", 0.20)),
            use_gradient=bool(getattr(self.params, "ncc_use_gradient", True)),
            grad_ksize=int(getattr(self.params, "ncc_grad_ksize", 3)),
            subpixel=bool(getattr(self.params, "ncc_subpixel", True)),
            max_cx_err_px=float(getattr(self.params, "lr_max_cx_err_px", 2.0)),
        )

        return mL_final, mR_final, boxesL, boxesR, pairs





