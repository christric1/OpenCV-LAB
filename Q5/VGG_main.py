from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
from VGG16 import Show_train_image, Print_parameter, Show_model, Show_chart, test


class Main_window(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('HW1')
        self.setStyleSheet("font: 12pt Arial")
        self.setFixedSize(300, 300)

        self.hBox = QHBoxLayout(self)

        self.groupBox1 = QGroupBox('VGG16_test')
        self.vBox1 = QVBoxLayout(self.groupBox1)
        self.groupBox1_btn1 = QPushButton("5.1 Show train images", self.groupBox1)
        self.groupBox1_btn2 = QPushButton("5.2 Show Hyperparameter", self.groupBox1)
        self.groupBox1_btn3 = QPushButton("5.3 Show model shortcut", self.groupBox1)
        self.groupBox1_btn4 = QPushButton("5.4 Show accuracy", self.groupBox1)

        self.lineEdit = QLineEdit()
        self.groupBox1_btn5 = QPushButton("5.5 Test", self.groupBox1)

        self.vBox1.addWidget(self.groupBox1_btn1, 1)
        self.vBox1.addWidget(self.groupBox1_btn2, 1)
        self.vBox1.addWidget(self.groupBox1_btn3, 1)
        self.vBox1.addWidget(self.groupBox1_btn4, 1)

        self.vBox1.addWidget(self.lineEdit, 1)
        self.vBox1.addWidget(self.groupBox1_btn5,1)

        self.hBox.addWidget(self.groupBox1, 1)

        self.groupBox1_btn1.clicked.connect(Show_train_image)
        self.groupBox1_btn2.clicked.connect(Print_parameter)
        self.groupBox1_btn3.clicked.connect(Show_model)
        self.groupBox1_btn4.clicked.connect(Show_chart)
        self.groupBox1_btn5.clicked.connect(self.for_test)

    def for_test(self):
        text = self.lineEdit.text()
        test(int(text))


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Main_window()
    ex.show()
    sys.exit(app.exec_())
