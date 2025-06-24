

import time

class Timer:
    def __init__(self, threshold_sec = 5.0):
        self.threshold = threshold_sec
        self.last_blink = time.time()

    def reset(self):
        self.last_blink = time.time()

    def check_elapsed(self):
        now = time.time()
        elapsed = now-self.last_blink
        return round(elapsed,3), elapsed > self.threshold