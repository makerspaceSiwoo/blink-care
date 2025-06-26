# BlinkCare 👁️

~~눈 깜빡임을 감지하여 장시간 집중 중 눈 건강을 보호하는 Python 기반 데스크탑 앱입니다.~~
MVP - 눈 깜빡임을 감지하는 Python, ML 기반 타이머

## 환경

- Windows 10

## 기능 ✨

- 눈을 감을 때마다 타이머 리셋, 5초를 넘기면 타이머 색깔 변화
- 실시간 카메라 추적 (MediaPipe + OpenCV)
- 눈 주변 마스킹 + 학습시킨 모델로 눈 깜빡임 감지
- 버전: v1.0.0

## 프로젝트 구조
'''
blink-care/
├── main.py                     # 데스크탑 blink-care 실행
├── blink_model_test.py         # ML 모델 테스트 및 작업 플로우 파일
├── requirements.txt
├── models/
│   └── blink_model.keras       # 학습된 ML 모델
├── modules/
│   ├── eye_crop.py             # mediapipe로 눈 부분 영역 이미지 추출
│   ├── preprocessor.py         # 흑백이미지로 변경 & resize
│   └── timer.py                # 눈 깜빡임 시간 측정
├── web/
│   ├── app.py                  # Streamlit/FastAPI 등 웹 인터페이스
│   └── static/                 # optional, html/css/js 등
├── scripts/
│   └── train.py                # 학습 스크립트
├── data/                       # (로컬에서 학습시 사용)
│   └── ...
├── __archive__/                # (더 이상 사용하지 않는 코드 - 중간 과정)
│   └── ...
└── README.md
'''

-  https://www.kaggle.com/datasets/arindamxd/eyes-open-closed-dataset/data - 학습에 사용한 데이터셋

