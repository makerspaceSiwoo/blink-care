import cv2
from modules.eye_crop import EyeCrop
from modules.preprocessor import preprocess_eye
from modules.timer import Timer
from tensorflow import keras

# 공통 설정
EYE_SIZE = (80, 80)
EYE_OFFSET = 30

def process_eye(frame, eye_side, eyecrop, model, timer):
    eye_img = eyecrop.crop_eye_area(frame, eye=eye_side)
    if eye_img is None:
        return

    input_tensor = preprocess_eye(eye_img)
    pred = model.predict(input_tensor, verbose=0)
    is_open = pred[0][0] > 0.5

    elapsed, is_over = timer.check_elapsed()
    if not is_open:
        timer.reset()

    # 이미지 위치 계산
    eye_small = cv2.resize(eye_img, EYE_SIZE)
    h, w, _ = frame.shape
    y2 = h - 10
    y1 = y2 - EYE_SIZE[1]

    if eye_side == "left":
        x2 = w - (EYE_SIZE[0] + 2 * EYE_OFFSET)
    else:  # right
        x2 = w - EYE_OFFSET
    x1 = x2 - EYE_SIZE[0]
    frame[y1:y2, x1:x2] = eye_small

    # 텍스트 출력
    label = f"{eye_side} : {'Open' if is_open else 'Closed'}"
    text_color = (0, 255, 0)
    timer_color = (0, 255, 0) if not is_over else (0, 0, 255)

    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{elapsed:.2f}s", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, timer_color, 1, cv2.LINE_AA)

def main_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    model = keras.models.load_model("model/blink_model.keras")
    eyecrop = EyeCrop()
    left_timer = Timer()
    right_timer = Timer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped_frame = cv2.flip(frame, 1)

        process_eye(flipped_frame, "left", eyecrop, model, left_timer)
        process_eye(flipped_frame, "right", eyecrop, model, right_timer)

        cv2.imshow("Monitor", flipped_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
