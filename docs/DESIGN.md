# DESIGN

## Потік даних
1. **IO** (`io.RTSPReader`) - кадри та таймштампи.
2. **SYNC** (`sync.best_time_aligned`) - вибирає L/R пари, близькі за часом.
3. **CALIBRATION** (`calibration.load_calibration`, `rectified_pair`) - ректифікація.
4. **DETECTOR** (`detector.StereoMotionDetector`) 
   - **MOTION** (`motion.DualBGModel`, `find_motion_bboxes`) - маска руху + бокси L/R.
   - **STEREO** (`stereo.gate_pairs_rectified`, `epipolar_ncc_match`) - паринг боксів.
5. **VISUALIZE** (`visualize.draw_boxes`, `stack_lr`) - відображення\малювання
6. **IO** (writer) - опціональний запис mp4.

## Конфігурування
Усі парметри зібрани до `config.SMDParams` та `config.AppConfig`. CLI парсит аргументи та збирає конфіги.

## Тестування
- Unit-тести для `motion.compute_mad`, `stereo.gate_pairs_rectified` та `epipolar_ncc_match`.
- Інтеграційні: мокові кадри L/R із синтетичними зсувами.
