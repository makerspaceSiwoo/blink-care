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

### 3. .exe 빌드 방법
pip install pyinstaller
pyinstaller main.py --onefile --windowed --name BlinkCare --add-data "D:\Projects\blink-care\venv\Lib\site-packages\mediapipe"





## 프로젝트 구조
blink-care/
├── main.py                    # 앱 실행 진입점
├── modules/
│ ├── constants.py             # 설정값 경로 및 기본값
│ ├── notifier.py              # PyQt 알림창
│ ├── blink_monitor.py         # 눈 감지 상태 + 타이머 추적
│ ├── eye_tracker.py           # Mediapipe로 눈 상태 감지
│ └── main_window.py           # 앱 실행 메인창
├── requirements.txt
├── .gitignore
└── README.md