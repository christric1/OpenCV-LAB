from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
import Camera_Calibration
import ResNet50


class Main_window(QWidget):
    def __init__(self):
        super(Main_window, self).__init__()

        self.setWindowTitle('HW1')
        self.setStyleSheet("font: 12pt Arial")

        self.hBox = QHBoxLayout(self)

        self.groupBox1 = QGroupBox('Calibration')
        self.vBox1 = QVBoxLayout(self.groupBox1)

        self.groupBox1_btn1 = QPushButton("2.1 Find Corners", self.groupBox1)
        self.groupBox1_btn2 = QPushButton("2.2 Find Intrinsic ", self.groupBox1)

        self.groupBox2 = QGroupBox("Find Extrinsic")
        self.vBox2 = QVBoxLayout(self.groupBox2)
        self.hBox2 = QHBoxLayout(self)
        self.label1 = QLabel("Select image : ", self.groupBox2)
        self.lineEdit = QLineEdit()
        self.groupBox2_btn1 = QPushButton("2.3 Find Extrinsic ", self.groupBox2)
        self.hBox2.addWidget(self.label1, 1)
        self.hBox2.addWidget(self.lineEdit, 1)
        self.vBox2.addLayout(self.hBox2, 1)
        self.vBox2.addWidget(self.groupBox2_btn1, 1)

        self.groupBox1_btn3 = QPushButton("2.4 Find Distortion", self.groupBox1)
        self.groupBox1_btn4 = QPushButton("2.5 Show Result", self.groupBox1)

        self.vBox1.addWidget(self.groupBox1_btn1, 1)
        self.vBox1.addWidget(self.groupBox1_btn2, 1)
        self.vBox1.addWidget(self.groupBox2, 1)
        self.vBox1.addWidget(self.groupBox1_btn3, 1)
        self.vBox1.addWidget(self.groupBox1_btn4, 1)

        self.groupBox3 = QGroupBox('ResNet50')
        self.vBox3 = QVBoxLayout(self.groupBox3)
        self.groupBox3_btn1 = QPushButton("5.1 Show model structure", self.groupBox1)
        self.groupBox3_btn2 = QPushButton("5.2 Show tensorBoard", self.groupBox1)
        self.groupBox3_btn3 = QPushButton("5.3 Test", self.groupBox1)
        self.lineEdit2 = QLineEdit()
        self.groupBox3_btn4 = QPushButton("5.4 Data Augmantation", self.groupBox1)

        self.vBox3.addWidget(self.groupBox3_btn1, 1)
        self.vBox3.addWidget(self.groupBox3_btn2, 1)
        self.vBox3.addWidget(self.groupBox3_btn3, 1)
        self.vBox3.addWidget(self.lineEdit2, 1)
        self.vBox3.addWidget(self.groupBox3_btn4, 1)

        self.hBox.addWidget(self.groupBox1, 1)
        self.hBox.addWidget(self.groupBox3, 1)

        # ----------------------------------------------------------------------

        self.CC = Camera_Calibration.Camera_Calibration()

        self.groupBox1_btn1.clicked.connect(self.CC.Find_corner)
        self.groupBox1_btn2.clicked.connect(self.CC.Find_Intrinsic)
        self.groupBox2_btn1.clicked.connect(lambda: self.CC.Find_Extrinsic(self.lineEdit.text()))
        self.groupBox1_btn3.clicked.connect(self.CC.Find_Distortion)
        self.groupBox1_btn4.clicked.connect(self.CC.Show_Undistorted)

        self.RN = ResNet50.ResNet50()

        self.groupBox3_btn1.clicked.connect(self.RN.Show_Model_Structure)
        self.groupBox3_btn2.clicked.connect(self.RN.Show_Tensorboard)
        self.groupBox3_btn3.clicked.connect(self.RN.Test)
        self.groupBox3_btn4.clicked.connect(self.RN.Data_Argument)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Main_window()
    ex.show()
    sys.exit(app.exec_())
