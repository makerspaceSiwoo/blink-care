import time
import cv2
from threading import Thread
from modules.eye_tracker import EyeTracker
from modules.constants import BLINK_THRESHOLD_SEC


class BlinkMonitor(Thread):
    def __init__(self, notifier, threshold_sec=BLINK_THRESHOLD_SEC):
        super().__init__()
        self.notifier = notifier
        self.threshold_sec = threshold_sec
        self.running = True
        self.last_blink_time = time.time()
        self.alert_shown = False

        self.tracker = EyeTracker()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.tracker.process(frame)
            now = time.time()

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    status = self.tracker.is_eye_open(frame, landmarks)
                    if not status["left"] or not status["right"]:
                        self.last_blink_time = now
                        if self.alert_shown:
                            self.alert_shown = False
                            self.notifier.hide_alert()
                    elif now - self.last_blink_time > self.threshold_sec:
                        if not self.alert_shown:
                            self.alert_shown = True
                            self.notifier.show_alert()

            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.cap.release()
