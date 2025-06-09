import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import time

# 디렉토리 경로
train_data_dir = "D:/Projects/blink-care/data/train"
test_data_dir = "D:/Projects/blink-care/data/test"
model_path = "D:/Projects/blink-care/models/blink_cnn.h5"

# 하이퍼파라미터
batch_size = 32
image_size = (64, 64)

# 이미지 불러오기 + 증강
train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
)

# 간단한 CNN 모델
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(*image_size, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 모델 저장
model.save(model_path)

# 성능 시각화
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.title("Training & Validation Performance")
plt.grid(True)
plt.show()


# 테스트 셋 정확도 추론 및 추론 속도 측정

# test_dir = "D:/Projects/blink-care/data/test"
# classes = ["open", "closed"]
# correct = 0
# total = 0
# times = []

# for label in classes:
#     dir_path = os.path.join(test_dir, label)
#     for fname in os.listdir(dir_path):
#         img_path = os.path.join(dir_path, fname)
#         img = image.load_img(img_path, target_size=image_size)
#         x = image.img_to_array(img) / 255.0
#         x = x.reshape((1,) + x.shape)

#         start = time.time()
#         pred = model.predict(x, verbose=0)[0][0]
#         end = time.time()

#         predicted_label = "open" if pred > 0.5 else "closed"
#         if predicted_label == label:
#             correct += 1
#         total += 1
#         times.append(end - start)

# print(f"정확도: {correct / total:.2%}")
# print(f"평균 추론 시간: {np.mean(times):.4f}초")
