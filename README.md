# Visual-radar

This project provides stereo camera calibration and real-time object detection using two RTSP network cameras.
It includes:

stereo_landscape_calibrate.py - captures synchronized image pairs, detects feature points, and generates calibration data (metric or projective mode).

stereo_rtsp_runtime.py - loads the calibration results, rectifies camera streams in real time, detects motion, matches objects between cameras, and (in metric mode) estimates distances.
