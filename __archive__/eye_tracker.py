import cv2
import mediapipe as mp
import numpy as np
from modules.constants import COLOR_DISTANCE_THRESHOLD, BRIGHTNESS_THRESHOLD


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
        self.last_debug_colors = {}

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)

    def is_eye_open(self, frame, landmarks):
        """
        눈 감지 알고리즘 : pupil 중심점의 색이 검은색이면 눈을 감은 것으로 판단
        """
        h, w, _ = frame.shape
        status = {}

        for eye, (pupil_idx, edge_idx) in self.pupil_check_pairs.items():
            p = landmarks.landmark[pupil_idx]

            pupil_cx = int(p.x * w)
            pupil_cy = int(p.y * h)

            if not (0 <= pupil_cx < w and 0 <= pupil_cy < h):
                status[eye] = False
                continue

            pupil_bgr = frame[pupil_cy, pupil_cx]

            # 디버깅 색상 저장
            self.last_debug_colors[f"{eye}_pupil"] = tuple(map(int, pupil_bgr))

            brightness = np.mean(pupil_bgr)
            status[eye] = brightness < BRIGHTNESS_THRESHOLD

        return status

    def is_eye_open_v2(self, frame, landmarks, radius=2):
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
            sclera_cx = int(
                (p.x + ratio * (e.x - p.x)) * w
            )  # sclera 계산 문제점 : 눈동자를 한쪽 측면으로 옮길 경우, 눈 감았다고 잘못 판단할 가능성 있음.
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

            # 디버깅 색상 저장
            self.last_debug_colors[f"{eye}_pupil"] = tuple(map(int, pupil_bgr))
            self.last_debug_colors[f"{eye}_sclera"] = tuple(map(int, sclera_bgr))

            color_distance = np.linalg.norm(
                np.array(pupil_bgr, dtype=np.float32)
                - np.array(sclera_bgr, dtype=np.float32)
            )
            status[eye] = color_distance > COLOR_DISTANCE_THRESHOLD

        return status

    def draw_debug_points(self, frame, landmarks=None):
        """
        - 오른쪽 상단에 pupil/sclera 색상 점 찍기
        - 눈 주변 landmark를 흰색 점으로 표시 (단 pupil 제외)
        """
        h, w, _ = frame.shape
        base_x = w - 40
        base_y = 40
        step_y = 60

        # 1. 디버깅 색상 점 (오른쪽 상단)
        for i, key in enumerate(
            ["left_pupil", "left_sclera", "right_pupil", "right_sclera"]
        ):
            color = self.last_debug_colors.get(key, (0, 0, 0))
            cv2.circle(frame, (base_x, base_y + i * step_y), 20, color, -1)
