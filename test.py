import subprocess, cv2, numpy as np

cmd = r'ffmpeg -rtsp_transport tcp -i "rtsp://admin:klop100500@192.168.1.125:554/cam/realmonitor?channel=1&subtype=0" -f mjpeg -q:v 4 -pix_fmt yuvj420p -'
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=10**7)

data = b''
while True:
    chunk = p.stdout.read(4096)
    if not chunk: break
    data += chunk
    i = data.find(b'\xff\xd8')  # SOI
    j = data.find(b'\xff\xd9')  # EOI
    if i!=-1 and j!=-1 and j>i:
        frame = cv2.imdecode(np.frombuffer(data[i:j+2], np.uint8), cv2.IMREAD_COLOR)
        data = data[j+2:]
        if frame is not None:
            cv2.imshow('test', frame)
            if cv2.waitKey(1) == 27: break