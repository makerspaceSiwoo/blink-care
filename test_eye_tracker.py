import cv2
from modules.eye_tracker import EyeTracker


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
tracker = EyeTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.process(frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            status = tracker.is_eye_open(frame, landmarks)
            tracker.draw_debug_points(frame, landmarks)

            left = "Open" if status["left"] else "Closed"
            right = "Open" if status["right"] else "Closed"
            cv2.putText(
                frame,
                    f"L: {left} / R: {right}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()