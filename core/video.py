import cv2
import time

class FPS:
    def __init__(self):
        self.last = time.time()
        self.fps = 0.0
    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps

def open_camera(index=0, width=1280, height=720):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
