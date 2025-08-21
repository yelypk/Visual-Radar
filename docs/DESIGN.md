# DESIGN

## Потік даних
1. **IO** (`open_stream`) → RTSP через OpenCV або FFmpeg→MJPEG, кадри+таймштампи.
2. **SYNC** (`best_time_aligned`) → підбір L/R пар близьких за часом.
3. **CALIBRATION** (`load_calibration`, `rectified_pair`) → ректифікація.
4. **DETECTOR** (`StereoMotionDetector`) →
   - **MOTION** (`DualBGModel`, `find_motion_bboxes`) → маска руху + бокси L/R.
   - **STEREO** (`gate_pairs_rectified`, `epipolar_ncc_match`) → паринг боксов.
5. **VISUALIZE** (`draw_boxes`, `resize_for_display`) → відображення.
6. **SNAPSHOTS** (`SnapshotSaver`) → збереження пар L/R з метаданими.
7. **IO(writer)** → опціональна запис в mp4.

## Конфіг
- `AppConfig` містить параметри стріму, відображення, знімків та SMD.
- Рідер: `opencv` або `ffmpeg_mjpeg` (транскодинг у MJPEG через ffmpeg).

## Маски фону
- `make_masks_static_and_slow` повертає дві маски:
  - статичний фон вирізаний (швидка різниця),
  - повільно-динамічний фон вирізаний (fast - slow, глушимо зони з великим slow).
