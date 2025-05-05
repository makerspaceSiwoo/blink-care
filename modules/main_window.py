import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget
from .notifier import Notifier
from .blink_monitor import BlinkMonitor


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blink Care")
        self.setFixedSize(300, 150)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.blink_monitor = None

        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)

    def start_monitoring(self):
        self.blink_monitor = BlinkMonitor(notifier=Notifier())
        self.blink_monitor.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_monitoring(self):
        if self.blink_monitor:
            self.blink_monitor.stop()
            self.blink_monitor.join()
            self.blink_monitor = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_monitoring()
        event.accept()
