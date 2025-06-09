import cv2
from threading import Thread


class Monitor(Thread):
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        self.running = False
        self.latest_frame = None

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.latest_frame = frame

    def get_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        self.capture.release()
