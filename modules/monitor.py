import cv2
from threading import Thread

# thread를 사용한 병렬 구조는 끊김이 있음 - 일단 제외하고 직렬 구조로. x 속도가 느린 이유는 dshow 때문


class Monitor:
    def __init__(self):
        self.capture = None
        self.thread = None
        self.running = False
        self.latest_fame = None

    def _update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.latest_fame = frame

    def start(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def get_frame(self):
        return self.latest_fame

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None

if __name__ == "__main__":
    monitor = Monitor()
    monitor.start()
    while True:
        frame = monitor.get_frame()
        if frame is None:
            continue
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    monitor.stop()