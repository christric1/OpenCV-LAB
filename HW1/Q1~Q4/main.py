from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys

import Image_Processing
import Image_Smoothing
import Edge_Detection
import Transforms


class Main_window(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('HW1')
        self.setStyleSheet("font: 12pt Arial")
        self.setFixedSize(960, 300)

        self.hBox = QHBoxLayout(self)

        self.groupBox1 = QGroupBox('Image Processing')
        self.vBox1 = QVBoxLayout(self.groupBox1)
        self.groupBox1_btn1 = QPushButton("1.1 Load Image", self.groupBox1)
        self.groupBox1_btn2 = QPushButton("1.2 Color Separation", self.groupBox1)
        self.groupBox1_btn3 = QPushButton("1.3 Color Transformation", self.groupBox1)
        self.groupBox1_btn4 = QPushButton("1.4 Blending", self.groupBox1)

        self.vBox1.addWidget(self.groupBox1_btn1, 1)
        self.vBox1.addWidget(self.groupBox1_btn2, 1)
        self.vBox1.addWidget(self.groupBox1_btn3, 1)
        self.vBox1.addWidget(self.groupBox1_btn4, 1)

        self.hBox.addWidget(self.groupBox1, 1)

        self.groupBox2 = QGroupBox('Image Smoothing')
        self.vBox2 = QVBoxLayout(self.groupBox2)
        self.groupBox2_btn1 = QPushButton("2.1 Gaussian blur", self.groupBox2)
        self.groupBox2_btn2 = QPushButton("2.2 Bilateral filter", self.groupBox2)
        self.groupBox2_btn3 = QPushButton("2.3 Median filter", self.groupBox2)

        self.vBox2.addWidget(self.groupBox2_btn1, 1)
        self.vBox2.addWidget(self.groupBox2_btn2, 1)
        self.vBox2.addWidget(self.groupBox2_btn3, 1)

        self.hBox.addWidget(self.groupBox2, 1)

        self.groupBox3 = QGroupBox('Edge Detection')
        self.vBox3 = QVBoxLayout(self.groupBox3)
        self.groupBox3_btn1 = QPushButton("3.1 Gaussian Blur", self.groupBox3)
        self.groupBox3_btn2 = QPushButton("3.2 Sobel X", self.groupBox3)
        self.groupBox3_btn3 = QPushButton("3.3 Sobel Y", self.groupBox3)
        self.groupBox3_btn4 = QPushButton("3.4 Magnitude", self.groupBox3)

        self.vBox3.addWidget(self.groupBox3_btn1, 1)
        self.vBox3.addWidget(self.groupBox3_btn2, 1)
        self.vBox3.addWidget(self.groupBox3_btn3, 1)
        self.vBox3.addWidget(self.groupBox3_btn4, 1)

        self.hBox.addWidget(self.groupBox3, 1)

        self.groupBox4 = QGroupBox('Transformation')
        self.vBox4 = QVBoxLayout(self.groupBox4)
        self.groupBox4_btn1 = QPushButton("4.1 Resize", self.groupBox4)
        self.groupBox4_btn2 = QPushButton("4.2 Translation", self.groupBox4)
        self.groupBox4_btn3 = QPushButton("4.3 Rotation, Scaling", self.groupBox4)
        self.groupBox4_btn4 = QPushButton("4.4 Shearing ", self.groupBox4)

        self.vBox4.addWidget(self.groupBox4_btn1, 1)
        self.vBox4.addWidget(self.groupBox4_btn2, 1)
        self.vBox4.addWidget(self.groupBox4_btn3, 1)
        self.vBox4.addWidget(self.groupBox4_btn4, 1)

        self.hBox.addWidget(self.groupBox4, 1)

        # ----------------------------------------------------------------------

        self.IP = Image_Processing.Image_Processing()

        self.groupBox1_btn1.clicked.connect(self.IP.load_image)
        self.groupBox1_btn2.clicked.connect(self.IP.color_separation)
        self.groupBox1_btn3.clicked.connect(self.IP.color_transformation)
        self.groupBox1_btn4.clicked.connect(self.IP.blending)

        self.IS = Image_Smoothing.Image_Smoothing()

        self.groupBox2_btn1.clicked.connect(self.IS.Gaussian_Blur)
        self.groupBox2_btn2.clicked.connect(self.IS.Bilateral_Filter)
        self.groupBox2_btn3.clicked.connect(self.IS.Median_Filter)

        self.ED = Edge_Detection.Edge_Detection()

        self.groupBox3_btn1.clicked.connect(self.ED.Gaussian_Blur)
        self.groupBox3_btn2.clicked.connect(self.ED.Sobel_X)
        self.groupBox3_btn3.clicked.connect(self.ED.Sobel_Y)
        self.groupBox3_btn4.clicked.connect(self.ED.Magnitude)

        self.T = Transforms.Transforms()

        self.groupBox4_btn1.clicked.connect(self.T.resize)
        self.groupBox4_btn2.clicked.connect(self.T.Translation)
        self.groupBox4_btn3.clicked.connect(self.T.Rotation_Scaling)
        self.groupBox4_btn4.clicked.connect(self.T.Shearing)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Main_window()
    ex.show()
    sys.exit(app.exec_())

