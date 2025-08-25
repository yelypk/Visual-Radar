# Visual-Radar

The project I am working on. 
Stereo surveillance on **OpenCV 4.x**. The app reads **two RTSP cameras**, time-syncs frames, rectifies them, detects motion, and **pairs L/R objects** along the epipolar line.

**Results**

<p float="left">
  <img src="./images/camera.jpg" width="49%">
  <img src="./images/cameraD.jpg" width="49%">
</p>

**Next development steps**

Align the two cameras with each other—i.e., estimate their relative orientation angle in 3D space. The cameras are mounted at the window, so this should be done using distant landmarks as references.