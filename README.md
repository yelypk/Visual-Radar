# Visual-radar
## How It Works
stereo_landscape_calibrate.py
│
├─ Capture synchronized image pairs from two RTSP cameras
│
├─ Detect & match features (SIFT / AKAZE / ORB)
│
├─ Estimate Fundamental (F) and, if intrinsics provided,
│ Essential (E), Rotation (R), Translation (t)
│
├─ Average results across pairs
│
├─ Save calibration data:
│ Metric mode → R_avg, t_dir_avg, rectification maps, Q
│ Projective mode → H1, H2 for rectification only
│
▼
stereo_rtsp_runtime.py
│
├─ Load calibration results (metric or projective)
│
├─ Open RTSP streams & rectify frames in real time
│
├─ Perform motion detection in each camera
│
├─ Match detected objects using stereo gating:
│ - Vertical alignment (Y tolerance)
│ - Disparity range filtering
│
├─ (Metric mode) Compute object distances in meters
│
▼
Display annotated stereo video with detections and distances
