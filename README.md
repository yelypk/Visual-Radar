# Visual-Radar

Stereo surveillance on **OpenCV 4.x**. The app reads **two RTSP cameras**, time-syncs frames, rectifies them, detects motion, and **pairs L/R objects** along the epipolar line.

## Features
- Inputs: **RTSP H.264** (OpenCV/FFmpeg backend) or **FFmpeg → MJPEG** pipe.
- L/R time synchronization (default tolerance 50 ms) with catch-up reads.
- Rectification with cached `remap` maps.
- Motion detection: dual background model (fast/slow), optional CLAHE, ROI and sky/water split.
- Stereo pairing: vertical gate + **NCC** (on gradients) with sub-pixel refinement.
- Tracking: lightweight IoU tracker with hysteresis.
- HUD overlay: FPS, L/R time delta, night/day flag, fail counters.
