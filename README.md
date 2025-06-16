# BlinkCare 👁️

눈 깜빡임을 감지하여 장시간 집중 중 눈 건강을 보호하는 Python 기반 데스크탑 앱입니다.

## 환경

- Windows 10

## 기능 ✨

- 눈을 5초 이상 깜빡이지 않으면 경고창 표시
- 실시간 카메라 추적 (MediaPipe + OpenCV)
- 버전: v1.0.0

## 사용법 🖥️

### 1. 설치

```bash
git clone https://github.com/yourname/blink-care.git
cd blink-care
python -m venv venv
venv\Scripts\activate    # 또는 source venv/bin/activate
pip install -r requirements.txt

### 2. 실행
python main.py

# ### 3. .exe 빌드 방법
# pip install pyinstaller
# pyinstaller main.py --onefile --windowed --name BlinkCare --add-data "D:\Projects\blink-care\venv\Lib\site-packages\mediapipe"





## 프로젝트 구조
blink-care/
├── main.py                     # 데스크탑 앱 진입점
├── blink_model_test.py         # ML 모델 테스트 및 작업 플로우 파일
├── test_eye_tracker.py         # 눈동자 색으로 눈 감음 여부 판단하는 테스트 파일
├── web_app.py                  # 웹 앱 실행 파일 (Streamlit 등)
├── requirements.txt
├── models/
│   └── blink_model.keras       # 학습된 ML 모델
├── modules/
│   ├── monitor.py              # 카메라 열고 프레임 추출
│   ├── eye_crop.py             # mediapipe로 눈 부분 영역 이미지 추출
│   ├── preprocessor.py         # 흑백이미지로 변경 & resize
│   ├── blink_classifier.py     # ML 모델 추론
│   └── blink_timer.py          # 눈 깜빡임 시간 측정
├── ui/
│   └── notifier.py             # PyQt 등 알림창 GUI
├── web/
│   ├── app.py                  # Streamlit/FastAPI 등 웹 인터페이스
│   └── static/                 # optional, html/css/js 등
├── scripts/
│   └── train.py                # 학습 스크립트
├── data/                       # (로컬에서 학습시 사용)
│   └── ...
└── README.md


# Thread 사용 - camera.py, blink_timer.py, main.py
#https://www.kaggle.com/datasets/arindamxd/eyes-open-closed-dataset/data - 학습에 사용한 데이터셋
```
