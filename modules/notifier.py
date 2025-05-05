from PyQt5 import QtWidgets, QtCore


class Notifier(QtWidgets.QWidget):
    show_signal = QtCore.pyqtSignal()
    hide_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blink Reminder")
        self.setFixedSize(280, 100)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("Please blink!", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 20px;
                background-color: rgba(255, 0, 0, 200);
                border-radius: 10px;
                padding: 20px;
            }
        """
        )
        layout.addWidget(self.label)

        # 시그널을 메인 스레드의 show/hide에 연결
        self.show_signal.connect(self.show)
        self.hide_signal.connect(self.hide)

    def show_alert(self):
        self.show_signal.emit()  # 메인스레드에서 show()

    def hide_alert(self):
        self.hide_signal.emit()  # 메인스레드에서 hide()
