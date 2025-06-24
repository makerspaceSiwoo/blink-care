import app as st
import cv2
from tensorflow import keras
from modules.eye_crop import EyeCrop
from modules.preprocessor import preprocess_eye
from modules.timer import Timer
import numpy as np

st.title("👁 눈 깜빡임 감지")
run = st.checkbox("카메라 시작")

frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    model = keras.models.load_model("model/blink_model.keras")
    eyecrop = EyeCrop()
    timer = Timer()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("카메라를 사용할 수 없습니다.")
            break

        flipped = cv2.flip(frame, 1)
        # (여기에 process_eye() 처리)
        frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb)

        if not st.session_state.get("run", True):
            break

    cap.release()
