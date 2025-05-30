import cv2        
cap = cv2.VideoCapture('/dev/v4l/by-id/usb-UGREEN_Camera_UGREEN_Camera_SN0001-video-index0', cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(30):
    ret, frame = cap.read()
import matplotlib.pyplot as plt
plt.imshow(frame)
plt.show()

"""
[
    [(381.9, 334.4), (0, 0)],  # center
    [(412.6, 310.8), (-0.3, 0)],  # up
    [(343.6, 363.6), (0.3, 0)],  # down
    [(326.3, 313.9), (0, -0.3)],  # left
    [(447.7, 358.8), (0, 0.3)],  # right
]
"""