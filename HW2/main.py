from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys


class Main_window(QWidget):
    def __init__(self):
        super(Main_window, self).__init__()

        self.setWindowTitle('HW1')
        self.setStyleSheet("font: 12pt Arial")
        self.setFixedSize(960, 300)

        self.hBox = QHBoxLayout(self)

        self.groupBox1 = QGroupBox('Calibration')
        self.vBox1 = QVBoxLayout(self.groupBox1)

        self.groupBox1_btn1 = QPushButton("2.1 Find Corners", self.groupBox1)
        self.groupBox1_btn2 = QPushButton("2.2 Find Intrinsic ", self.groupBox1)

        self.groupBox2 = QGroupBox("Find Extrinsic")

        self.vBox1.addWidget(self.groupBox1_btn1, 1)
        self.vBox1.addWidget(self.groupBox1_btn2, 1)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Main_window()
    ex.show()
    sys.exit(app.exec_())