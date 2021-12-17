import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


class Image_Smoothing:

    def __init__(self):
        self.whiteNoise_img = cv2.imread("../data/Q2_Image/Lenna_whiteNoise.jpg")
        self.pepperSalt_img = cv2.imread("../data/Q2_Image/Lenna_pepperSalt.jpg")

    def Gaussian_Blur(self):
        result_img = cv2.GaussianBlur(self.whiteNoise_img, (5, 5), 0)
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title("whiteNoise_img")
        Lenna_whiteNoise_img = cv2.cvtColor(self.whiteNoise_img, cv2.COLOR_BGR2RGB)
        plt.imshow(Lenna_whiteNoise_img)

        plt.subplot(1, 2, 2)
        plt.title("whiteNoise_Gaussian_img")
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        plt.imshow(result_img)

        plt.show()

    def Bilateral_Filter(self):

        result_img = cv2.bilateralFilter(self.whiteNoise_img, 9, 90, 90)
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title("whiteNoise_img")
        Lenna_whiteNoise_img = cv2.cvtColor(self.whiteNoise_img, cv2.COLOR_BGR2RGB)
        plt.imshow(Lenna_whiteNoise_img)

        plt.subplot(1, 2, 2)
        plt.title("whiteNoise_Bilateral_img")
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        plt.imshow(result_img)

        plt.show()

    def Median_Filter(self):

        result_3x3_img = cv2.medianBlur(self.pepperSalt_img, 3)
        result_5x5_img = cv2.medianBlur(self.pepperSalt_img, 5)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("whiteNoise_img")
        Lenna_pepperSalt_img = cv2.cvtColor(self.pepperSalt_img, cv2.COLOR_BGR2RGB)
        plt.imshow(Lenna_pepperSalt_img)

        plt.subplot(1, 3, 2)
        plt.title("whiteNoise_Median_3x3_img")
        result_3x3_img = cv2.cvtColor(result_3x3_img, cv2.COLOR_BGR2RGB)
        plt.imshow(result_3x3_img)

        plt.subplot(1, 3, 3)
        plt.title("whiteNoise_Median_5x5_img")
        result_5x5_img = cv2.cvtColor(result_5x5_img, cv2.COLOR_BGR2RGB)
        plt.imshow(result_5x5_img)

        plt.show()