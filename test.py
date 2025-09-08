import numpy as np
d = np.load("calib/intrinsics_stereo.npz", allow_pickle=True)
for k in d.files:
    a = d[k];  a.shape  # попытка чтения (важно)
    print(k, "OK", a.shape, a.dtype)