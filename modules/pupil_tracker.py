import cv2
import mediapipe as mp

class BlinkTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # 눈동자 랜드마크 활성화
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 눈동자 중심점 인덱스만
        self.iris_centers = {
            # 왼쪽 눈
            "left_center": 468,    # 왼쪽 눈동자 중심
            # "left_top": 386,       # 왼쪽 윗눈꺼풀 (눈 위)
            # "left_bottom": 374,    # 왼쪽 아랫눈꺼풀 (눈 아래)
            # "left_outer": 33,      # 왼쪽 눈 바깥쪽 끝
            # "left_inner": 133,     # 왼쪽 눈 안쪽 끝
            # "left_upper_mid": 160, # 왼쪽 눈 윗눈꺼풀 중앙
            # "left_lower_mid": 144, # 왼쪽 눈 아랫눈꺼풀 중앙

            # 오른쪽 눈
            "right_center": 473,    # 오른쪽 눈동자 중심
            # "right_top": 159,       # 오른쪽 윗눈꺼풀 (눈 위)
            # "right_bottom": 145,    # 오른쪽 아랫눈꺼풀 (눈 아래)
            # "right_outer": 263,     # 오른쪽 눈 바깥쪽 끝
            # "right_inner": 362,     # 오른쪽 눈 안쪽 끝
            # "right_upper_mid": 387, # 오른쪽 눈 윗눈꺼풀 중앙
            # "right_lower_mid": 373, # 오른쪽 눈 아랫눈꺼풀 중앙
        }

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results

    def draw_iris_centers(self, frame, landmarks):
        h, w, _ = frame.shape

        for eye, idx in self.iris_centers.items():
            point = landmarks.landmark[idx]
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # 초록색 점 찍기

            # 눈동자가 정상적으로 검출됐을 때만 점 찍기
            if 0 < point.x < 1 and 0 < point.y < 1:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = BlinkTracker()

    if not cap.isOpened():
        print("Camera not accessible")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = tracker.process(frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                tracker.draw_iris_centers(frame, landmarks)

        cv2.imshow('Blink Tracker', frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 키
            break

    cap.release()
    cv2.destroyAllWindows()
