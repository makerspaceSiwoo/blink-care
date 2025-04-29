import cv2
import mediapipe as mp
import numpy as np

class BlinkTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pupil_check_pairs = {
            "left": (468, 33),
            "right": (473, 263),
        }

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results

    def is_eye_open(self, frame, landmarks, radius=3):
        h, w, _ = frame.shape
        status = {}

        for eye, (pupil_idx, edge_idx) in self.pupil_check_pairs.items():
            p = landmarks.landmark[pupil_idx]
            e = landmarks.landmark[edge_idx]

            # pupil 좌표
            pupil_cx = int(p.x * w)
            pupil_cy = int(p.y * h)

            # 중간 좌표 (pupil ↔ eye corner 중간)
            mid_cx = int(((p.x + e.x) / 2) * w)
            mid_cy = int(((p.y + e.y) / 2) * h)

            # 좌표 유효성 체크
            if not (0 <= pupil_cx < w and 0 <= pupil_cy < h and 0 <= mid_cx < w and 0 <= mid_cy < h):
                status[eye] = False
                continue

            # pupil 지점 색상
            pupil_bgr = frame[pupil_cy, pupil_cx]

            # 디버깅용 점 찍기
            # cv2.circle(frame, (pupil_cx, pupil_cy), 3, (0, 255, 0), -1)  # 초록: pupil

            debug_x = w - 100 if eye == "left" else w - 100
            debug_y = 100 if eye == "left" else 200
            cv2.circle(frame, (debug_x, debug_y), 30, color=tuple(map(int, pupil_bgr)), thickness=-1)


            # cv2.circle(frame, (mid_cx, mid_cy), radius, (0, 255, 255), 1)  # 노랑: 중간

            # 중간 지점 주변 작은 원 마스크
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (mid_cx, mid_cy), radius, 255, -1)

            masked_pixels = cv2.bitwise_and(frame, frame, mask=mask)
            bgr_pixels = masked_pixels[np.where(mask == 255)]

            if len(bgr_pixels) == 0:
                status[eye] = False
                continue

            # 흰자 영역 평균 색
            sclera_bgr = np.mean(bgr_pixels, axis=0)
            
            # sclera 디버깅 점 (오른쪽에)
            cv2.circle(frame, (debug_x - 80, debug_y), 30, color=tuple(map(int, sclera_bgr)), thickness=-1)

            # pupil 색과 sclera 평균 색 차이 계산
            color_distance = np.linalg.norm(np.array(pupil_bgr, dtype=np.float32) - np.array(sclera_bgr, dtype=np.float32))

            cv2.putText(frame, f"{eye} dist: {int(color_distance)}", (30, 60 if eye == "left" else 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


            # 차이가 크면 눈 뜬 것으로 간주
            if color_distance > 40:  # threshold 40 정도 (필요시 조정)
                status[eye] = True
            else:
                status[eye] = False

        return status

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = BlinkTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = tracker.process(frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                status = tracker.is_eye_open(frame, landmarks)

                # 결과 표시
                left = "Open" if status["left"] else "Closed"
                right = "Open" if status["right"] else "Closed"
                cv2.putText(frame, f"L: {left} / R: {right}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Blink Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
