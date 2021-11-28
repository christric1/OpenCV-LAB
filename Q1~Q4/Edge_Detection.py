import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math


class Edge_Detection:
    def __init__(self):
        self.img = cv2.imread("../data/Q3_Image/House.jpg")

    def Gaussian_Blur(self):
        # 3*3 Gaussian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))

        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        grad = signal.convolve2d(gray_img, gaussian_kernel, boundary='symm', mode='same')  # 卷積

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("origin_img")
        origin_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.imshow(origin_img)

        plt.subplot(1, 3, 2)
        plt.title("gray_img")
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
        plt.imshow(gray_img)

        plt.subplot(1, 3, 3)
        plt.title("Gaussian_gray_img")
        plt.imshow(grad, cmap=plt.get_cmap('gray'))

        plt.show()

    def Sobel_X(self):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        grad = signal.convolve2d(gray_img, gaussian_kernel, boundary='symm', mode='same')  # 卷積

        sobel_x_img = signal.convolve2d(grad, sobel_x, boundary='symm', mode='same')
        sobel_x_img = np.abs(sobel_x_img)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Gaussian_gray_img")
        plt.imshow(gray_img, cmap=plt.get_cmap('gray'))

        plt.subplot(1, 2, 2)
        plt.title("SobelX_Gaussian_gray_img")
        plt.imshow(sobel_x_img, cmap=plt.get_cmap('gray'))

        plt.show()

    def Sobel_Y(self):

        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        grad = signal.convolve2d(gray_img, gaussian_kernel, boundary='symm', mode='same')  # 卷積

        sobel_y_img = signal.convolve2d(grad, sobel_y, boundary='symm', mode='same')
        sobel_y_img = np.abs(sobel_y_img)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Gaussian_gray_img")
        plt.imshow(gray_img, cmap=plt.get_cmap('gray'))

        plt.subplot(1, 2, 2)
        plt.title("SobelY_Gaussian_gray_img")
        plt.imshow(sobel_y_img, cmap=plt.get_cmap('gray'))

        plt.show()

    def Magnitude(self):

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        grad = signal.convolve2d(gray_img, gaussian_kernel, boundary='symm', mode='same')  # 卷積

        sobel_x_img = signal.convolve2d(grad, sobel_x, boundary='symm', mode='same')
        sobel_x_img = np.abs(sobel_x_img)

        sobel_y_img = signal.convolve2d(grad, sobel_y, boundary='symm', mode='same')
        sobel_y_img = np.abs(sobel_y_img)

        magnitude = np.sqrt(sobel_y_img ** 2 + sobel_x_img ** 2)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("SobelX_Gaussian_gray_img")
        plt.imshow(sobel_x_img, cmap=plt.get_cmap('gray'))

        plt.subplot(1, 3, 2)
        plt.title("SobelY_Gaussian_gray_img")
        plt.imshow(sobel_y_img, cmap=plt.get_cmap('gray'))

        plt.subplot(1, 3, 3)
        plt.title("magnitude_img")
        plt.imshow(magnitude, cmap=plt.get_cmap('gray'))

        plt.show()
