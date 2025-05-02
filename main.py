import cv2
import time
import threading
import PySimpleGUI as sg
from modules.blink_tracker import BlinkTracker

# 전역 메시지창
alert_window = None

# 블로킹 메시지창 열기
def show_blink_alert():
    global alert_window
    if alert_window is None:
        layout = [[sg.Text("눈을 깜빡이세요!", font=("Arial", 16), justification='center')]]
        alert_window = sg.Window("눈 건강 알림", layout,
                                 finalize=True, keep_on_top=True,
                                 no_titlebar=True, grab_anywhere=True)

# 메시지창 닫기
def close_blink_alert():
    global alert_window
    if alert_window is not None:
        alert_window.close()
        alert_window = None

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = BlinkTracker()

    last_blink_time = time.time()
    alert_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = tracker.process(frame)
        now = time.time()

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                status = tracker.is_eye_open(frame, landmarks)

                # 눈을 감으면 타이머 리셋 + 창 닫기
                if not status["left"] or not status["right"]:
                    last_blink_time = now
                    if alert_active:
                        close_blink_alert()
                        alert_active = False

                # 눈을 일정 시간(5초) 이상 안 감았으면 경고
                elif now - last_blink_time > 1:
                    if not alert_active:
                        show_blink_alert()
                        alert_active = True

        # OpenCV 창에 눈 상태 보여주기
        cv2.imshow("Blink Care", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    close_blink_alert()
    cap.release()
    cv2.destroyAllWindows()