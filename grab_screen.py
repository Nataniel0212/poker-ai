"""Capture screen and save as image so we can analyze positions."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()
os.chdir(_dir)

import mss
import numpy as np
import cv2

with mss.mss() as sct:
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    os.makedirs(os.path.join(_dir, "debug_images"), exist_ok=True)
    path = os.path.join(_dir, "debug_images", "screen_capture.png")
    success, buf = cv2.imencode('.png', frame_bgr)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
    print(f"Screen captured: {monitor['width']}x{monitor['height']}")
    print(f"Saved to: {path}")
