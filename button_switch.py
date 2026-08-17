import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QFont, QBrush, QColor, QPen


class SwitchButton(QWidget):
    def __init__(self, parent=None):
        super(SwitchButton, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # self.resize(70, 30)
        # SwitchButtonstate：True is ON，False is OFF
        self.state = False
        self.setFixedSize(60, 30)

    def mousePressEvent(self, event):
        '''
        set click event for state change
        '''
        super(SwitchButton, self).mousePressEvent(event)
        self.state = False if self.state else True
        self.update()

    def paintEvent(self, event):
        '''Set the button'''
        super(SwitchButton, self).paintEvent(event)

        # Create a renderer and set anti-aliasing and smooth transitions
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        # Defining font styles
        font = QFont("Arial")
        font.setPixelSize(self.height() // 3)
        painter.setFont(font)
        # SwitchButton state：ON
        if self.state:
            # Drawing background
            painter.setPen(Qt.NoPen)
            brush = QBrush(QColor('#bd93f9'))
            painter.setBrush(brush)
            # Top left corner of the rectangle coordinate
            rect_x = 0
            rect_y = 0
            rect_width = self.width()
            rect_height = self.height()
            rect_radius = self.height() // 2
            painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, rect_radius, rect_radius)
            # Drawing slides circle
            painter.setPen(Qt.NoPen)
            brush.setColor(QColor('#ffffff'))
            painter.setBrush(brush)
            # Phase difference pixel point
            # Top left corner of the rectangle coordinate
            diff_pix = 3
            rect_x = self.width() - diff_pix - (self.height() - 2 * diff_pix)
            rect_y = diff_pix
            rect_width = (self.height() - 2 * diff_pix)
            rect_height = (self.height() - 2 * diff_pix)
            rect_radius = (self.height() - 2 * diff_pix) // 2
            painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, rect_radius, rect_radius)

            # ON txt set
            painter.setPen(QPen(QColor('#ffffff')))
            painter.setBrush(Qt.NoBrush)
            painter.drawText(QRect(int(self.height() / 3), int(self.height() / 3.5), 80, 80), Qt.AlignLeft, 'ON')
        # SwitchButton state：OFF
        else:
            # Drawing background
            painter.setPen(Qt.NoPen)
            brush = QBrush(QColor('#525555'))
            painter.setBrush(brush)
            # Top left corner of the rectangle coordinate
            rect_x = 0
            rect_y = 0
            rect_width = self.width()
            rect_height = self.height()
            rect_radius = self.height() // 2
            painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, rect_radius, rect_radius)

            # Drawing slides circle
            pen = QPen(QColor('#999999'))
            pen.setWidth(1)
            painter.setPen(pen)
            # Phase difference pixel point
            diff_pix = 3
            # Top left corner of the rectangle coordinate
            rect_x = diff_pix
            rect_y = diff_pix
            rect_width = (self.height() - 2 * diff_pix)
            rect_height = (self.height() - 2 * diff_pix)
            rect_radius = (self.height() - 2 * diff_pix) // 2
            painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, rect_radius, rect_radius)

            # OFF txt set
            painter.setBrush(Qt.NoBrush)
            painter.drawText(QRect(int(self.width() * 1 / 2), int(self.height() / 3.5), 50, 20), Qt.AlignLeft, 'OFF')


def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setGeometry(100, 100, 100, 290)
    window.setWindowTitle("Switch Button Example")
    switch1 = SwitchButton()
    switch2 = SwitchButton()
    layout = QVBoxLayout()
    layout.addWidget(switch1)
    layout.addWidget(switch2)
    window.setCentralWidget(QWidget())
    window.centralWidget().setLayout(layout)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()