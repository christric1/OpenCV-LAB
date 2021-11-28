import cv2
import numpy as np
import matplotlib.pyplot as plt


class Image_Processing:

    def __init__(self):
        self.img = cv2.imread("../data/Q1_Image/Sun.jpg")

    def load_image(self):
        cv2.imshow('My Image', self.img)
        print("Height : %d\nWidth : %d" % (self.img.shape[0], self.img.shape[1]))

    def color_separation(self):
        B, G, R = cv2.split(self.img)
        z = np.zeros([self.img.shape[0], self.img.shape[1]], np.uint8)

        B_img = cv2.merge([B, z, z])
        G_img = cv2.merge([z, G, z])
        R_img = cv2.merge([z, z, R])

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(B_img)

        plt.subplot(1, 3, 2)
        plt.imshow(G_img)

        plt.subplot(1, 3, 3)
        plt.imshow(R_img)

        plt.show()

    def color_transformation(self):

        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        B, G, R = cv2.split(self.img)
        average_img = np.floor((B.astype("uint16") + G.astype("uint16") + R.astype("uint16")) / 3).astype("uint8")

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title("gray_img")
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
        plt.imshow(gray_img)

        plt.subplot(1, 2, 2)
        plt.title("average_img")
        average_img = cv2.cvtColor(average_img, cv2.COLOR_BGR2RGB)
        plt.imshow(average_img)

        plt.show()

    def blending(self):

        cv2.namedWindow('image')
        cv2.createTrackbar('alpha', 'image', 0, 256, (lambda x: None))

        img1 = cv2.imread("../data/Q1_Image/Dog_Strong.jpg")
        img2 = cv2.imread("../data/Q1_Image/Dog_Weak.jpg")

        while True:
            alpha = cv2.getTrackbarPos("alpha", 'image') / 256
            beta = 1 - alpha

            output = cv2.addWeighted(img1, alpha, img2, beta, 0)

            cv2.imshow("image", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
