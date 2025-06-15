import cv2
import numpy as np


def preprocess_eye(eye_img: np.ndarray) -> np.ndarray:
    # 1. 흑백 변환
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

    # 2. 크기 조정
    resized = cv2.resize(gray, (64, 64))

    # 3. 정규화
    normalized = resized.astype(np.float32) / 255.0

    # 4. 모델 입력 형태로 변형
    input_tensor = normalized.reshape(1, 64, 64, 1)
    return input_tensor
