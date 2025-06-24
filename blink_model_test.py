import cv2
from modules.eye_crop import EyeCrop
from modules.preprocessor import preprocess_eye
from modules.timer import Timer
from tensorflow import keras

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

eyecrop = EyeCrop()
model = keras.models.load_model("model/blink_model.keras")

left_timer = Timer()
right_timer = Timer()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flipped_frame = cv2.flip(frame, 1)

    # 왼쪽 눈 처리
    left_eye = eyecrop.crop_eye_area(flipped_frame, eye="left")
    if left_eye is not None:
        input_tensor = preprocess_eye(left_eye)
        pred = model.predict(input_tensor, verbose=0)
        is_open = pred[0][0] > 0.5


        elapsed, is_over = left_timer.check_elapsed()

        if not is_open : 
            left_timer.reset()

        left_eye_small = cv2.resize(left_eye, (80, 80))
        h, w, _ = flipped_frame.shape
        y2 = h - 10
        y1 = y2 - 80
        x2 = w - 20 - 80 - 20  # 오른쪽 눈 옆에 공간
        x1 = x2 - 80
        flipped_frame[y1:y2, x1:x2] = left_eye_small

        # 1. 텍스트 내용
        label = "left : Open" if is_open else "left : Closed"
        # 2. 텍스트 위치 (왼쪽 눈 미리보기 위쪽)
        text_pos = (x1, y1 - 5)  # 이미지 위 살짝 위에 표시
        # 3. 텍스트 그리기
        cv2.putText(
            flipped_frame,
            label,
            text_pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),  # 초록색
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # 4. timer 그리기
        timer_color = (0,255,0) if not is_over else (0,0,255)

        cv2.putText(
            flipped_frame,
            f"{elapsed:.2f}s",
            (x1, y1 - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=timer_color,
            thickness=1,
            lineType=cv2.LINE_AA
        )


    # 오른쪽 눈 처리
    right_eye = eyecrop.crop_eye_area(flipped_frame, eye="right")
    if right_eye is not None:
        input_tensor = preprocess_eye(right_eye)
        pred = model.predict(input_tensor, verbose=0)
        is_open = pred[0][0] > 0.5

        elapsed, is_over = right_timer.check_elapsed()

        if not is_open : 
            right_timer.reset()

        right_eye_small = cv2.resize(right_eye, (80, 80))
        h, w, _ = flipped_frame.shape
        y2 = h - 10
        y1 = y2 - 80
        x2 = w - 20
        x1 = x2 - 80
        flipped_frame[y1:y2, x1:x2] = right_eye_small

        # 1. 텍스트 내용
        label = "right : Open" if is_open else "right : Closed"
        # 2. 텍스트 위치 (왼쪽 눈 미리보기 위쪽)
        text_pos = (x1, y1 - 5)  # 이미지 위 살짝 위에 표시
        # 3. 텍스트 그리기
        cv2.putText(
            flipped_frame,
            label,
            text_pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),  # 초록색
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # 4. timer 그리기
        timer_color = (0,255,0) if not is_over else (0,0,255)

        cv2.putText(
            flipped_frame,
            f"{elapsed:.2f}s",
            (x1, y1 - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=timer_color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    cv2.imshow("Monitor", flipped_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
