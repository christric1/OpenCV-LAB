import cv2
import glob
import numpy as np


class Camera_Calibration:
    def __init__(self):

        print("123")

    def Find_corner(self):

        images = glob.glob('data\\Q2_Image\\*.bmp')

        for idx, image_name in enumerate(images):

            img = cv2.imread(image_name)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_img, (8, 11), None)

            if ret:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (8, 11), corners, ret)
                img = cv2.resize(img, (800, 800))
                cv2.imshow('img', img)
                cv2.waitKey(500)

    def Find_Intrinsic(self):

        # set the nx, ny according the calibration chessboard pictures.
        nx = 8
        ny = 11

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...(6,5,0)
        objP = np.zeros((nx * ny, 3), np.float32)
        objP[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objPoints = []  # 3d points in real world space
        imgPoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('data\\Q2_Image\\*.bmp')

        # Step through the list and search for chessboard corners
        for idx, image_name in enumerate(images):

            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret:
                objPoints.append(objP)
                imgPoints.append(corners)

        # Get image size
        img = cv2.imread(images[0])
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img_size, None, None)

        print("Intrinsic : ")
        print(mtx)

    def Find_Extrinsic(self):


if __name__ == '__main__':

    d = Camera_Calibration()
    d.Find_corner()
