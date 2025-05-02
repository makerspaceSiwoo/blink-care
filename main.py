import cv2
import time
import numpy as np
from modules.blink_tracker import BlinkTracker

# 메시지창 이미지 생성 (OpenCV)
def show_opencv_alert():
    msg_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(msg_img, "Blink your eyes!", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("ALERT", msg_img)

def close_opencv_alert():
    cv2.destroyWindow("ALERT")

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
                    print("eyes closed")
                    last_blink_time = now
                    if alert_active:
                        close_opencv_alert()
                        alert_active = False

                # 눈 뜨고 있는 시간 5초 초과 → 경고창 표시
                elif now - last_blink_time > 4:
                    if not alert_active:
                        show_opencv_alert()
                        alert_active = True

                # 상태 유지 시, 매 프레임마다 alert 창 갱신
                if alert_active:
                    show_opencv_alert()

        # 프레임 표시
        cv2.imshow("Blink Care", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    close_opencv_alert()
    cap.release()
    cv2.destroyAllWindows()