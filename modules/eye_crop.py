import cv2
from typing import Literal
import mediapipe as mp


# 왼쪽 눈 : 위-159, 아래-145, 바깥-33, 안쪽-133, 눈동자 468
# 오른쪽 눈 : 위-386, 아래-274, 바깥-263, 안쪽-362, 눈동자:473


class EyeCrop:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pupils = {"left": 468, "right": 473}
        self.eye_indices = {"left": [33, 133, 159, 145], "right": [263, 362, 386, 374]}

    def crop_eye_area(self, frame, eye: Literal["left", "right"] = "left"):
        """
        이미지에서 눈 주위 정사각형 영역을 잘라서 리턴

        parameters:
            frame (np.ndarray) : OpenCV 프레임
            eye (str) : "left" or "right:

        returns:
            cropped (np.ndarray) : 눈 영역 이미지
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # 눈 주위 좌표 추출
        coords = []
        for idx in self.eye_indices[eye]:
            lm = face_landmarks.landmark[idx]
            coords.append((int(lm.x * w), int(lm.y * h)))

        # 눈동자 중심
        pupil = face_landmarks.landmark[self.pupils[eye]]
        cx, cy = int(pupil.x * w), int(pupil.y * h)

        # 눈 주위 길이 측정 후 눈동자 중심으로 적당한 영역 추출
        xs, ys = zip(*coords)
        eye_width = max(max(xs) - min(xs), max(ys) - min(ys))
        percent = 0.8

        x1, y1 = max(0, cx - int(eye_width * percent)), max(
            0, cy - int(eye_width * percent)
        )
        x2, y2 = min(w, cx + int(eye_width * percent)), min(
            h, cy + int(eye_width * percent)
        )

        return frame[y1:y2, x1:x2].copy()
