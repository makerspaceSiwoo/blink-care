import cv2
import mediapipe as mp
import numpy as np
from .constants import COLOR_DISTANCE_THRESHOLD


class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pupil_check_pairs = {
            "right": (468, 33),
            "left": (473, 263),
        }

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)

    def is_eye_open(self, frame, landmarks, radius=2):
        """
        눈 감지 알고리즘 : pupil 중심점과 edge 사이를 sclera로 정의한 후,
        sclera 샘플링 위치에서 눈동자 색상과 비교하여 눈 감음 여부 판단
        """
        h, w, _ = frame.shape
        status = {}

        for eye, (pupil_idx, edge_idx) in self.pupil_check_pairs.items():
            p = landmarks.landmark[pupil_idx]
            e = landmarks.landmark[edge_idx]

            pupil_cx = int(p.x * w)
            pupil_cy = int(p.y * h)

            ratio = 0.5
            sclera_cx = int((p.x + ratio * (e.x - p.x)) * w)
            sclera_cy = int((p.y + ratio * (e.y - p.y)) * h)

            if not (
                0 <= pupil_cx < w
                and 0 <= pupil_cy < h
                and 0 <= sclera_cx < w
                and 0 <= sclera_cy < h
            ):
                status[eye] = False
                continue

            pupil_bgr = frame[pupil_cy, pupil_cx]

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (sclera_cx, sclera_cy), radius, 255, -1)
            masked_pixels = cv2.bitwise_and(frame, frame, mask=mask)
            bgr_pixels = masked_pixels[np.where(mask == 255)]

            if len(bgr_pixels) == 0:
                status[eye] = False
                continue

            sclera_bgr = np.mean(bgr_pixels, axis=0)
            color_distance = np.linalg.norm(
                np.array(pupil_bgr, dtype=np.float32)
                - np.array(sclera_bgr, dtype=np.float32)
            )
            status[eye] = color_distance > COLOR_DISTANCE_THRESHOLD

        return status


if __name__ == "__main__":
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
