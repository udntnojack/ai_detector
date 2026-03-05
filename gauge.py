from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtCore import Qt


class GaugeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.value = 0  # 0–100
        self.setMinimumSize(150, 150)
        self.setMaximumSize(150, 150)

    def setValue(self, val):
        self.value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)

        # background circle
        pen = QPen(QColor("#dddddd"), 12)
        painter.setPen(pen)
        painter.drawArc(rect, 225 * 16, -270 * 16)

        # value arc
        pen = QPen(QColor("#ff4444"), 12)
        painter.setPen(pen)
        span = int(-270 * (self.value / 100))
        painter.drawArc(rect, 225 * 16, span * 16)

        # text
        painter.setPen(Qt.black)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.value}%")