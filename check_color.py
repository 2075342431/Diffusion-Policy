import cv2
import numpy as np

vid = cv2.VideoCapture("data/lerobot_pickcube_997/videos/observation.images.base_camera/chunk-000/file-000.mp4")
ret, frame = vid.read()
if ret:
    # frame is BGR if read by cv2.VideoCapture
    # Let's find the brightest red or blue pixel
    # In BGR, channel 0 is Blue, 1 is Green, 2 is Red
    b, g, r = cv2.split(frame)
    print(f"Max B: {b.max()}, Max R: {r.max()}")
    
    # Check if there are strong blue pixels vs strong red pixels
    # The PickCube has a Red cube and Green goal.
    # If the video was saved with RGB data directly into VideoWriter without cvtColor,
    # the VideoWriter treats it as BGR. Then when VideoCapture reads it, it returns it.
    # Actually, let's just see if Max R > Max B. If the cube is red, there should be bright red.
    print(f"Mean R of brightest red pixels: {r[r > 150].mean() if len(r[r>150]) > 0 else 0}")
    print(f"Mean B of brightest blue pixels: {b[b > 150].mean() if len(b[b>150]) > 0 else 0}")
