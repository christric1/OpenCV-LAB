import cv2
import numpy as np


class Transforms:
    def __init__(self):

        self.img = cv2.imread("../data/Q4_Image/SQUARE-01.png")

    def resize(self):
        resize_img = cv2.resize(self.img, (256, 256))
        cv2.imshow("resize_img", resize_img)

    def Translation(self):

        resize_img = cv2.resize(self.img, (256, 256))

        M = np.float32([[1, 0, 0], [0, 1, 60]])
        dst = cv2.warpAffine(resize_img, M, (400, 300))

        cv2.imshow("translation", dst)

    def Rotation_Scaling(self):

        resize_img = cv2.resize(self.img, (256, 256))

        M = np.float32([[1, 0, 0], [0, 1, 60]])
        dst = cv2.warpAffine(resize_img, M, (400, 300))

        angle = 10
        rotate_mat = cv2.getRotationMatrix2D((128, 188), angle, 0.5)

        dst = cv2.warpAffine(dst, rotate_mat, (400, 300))

        cv2.imshow("Rotation_Scaling", dst)

    def Shearing(self):

        resize_img = cv2.resize(self.img, (256, 256))

        M = np.float32([[1, 0, 0], [0, 1, 60]])

        dst = cv2.warpAffine(resize_img, M, (400, 300))

        angle = 10
        rotate_mat = cv2.getRotationMatrix2D((128, 188), angle, 0.5)

        dst = cv2.warpAffine(dst, rotate_mat, (400, 300))

        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        H = cv2.getAffineTransform(pts1, pts2)

        dst = cv2.warpAffine(dst, H, (400, 300))

        cv2.imshow("translation", dst)
