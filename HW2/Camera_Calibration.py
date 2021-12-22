import cv2
import glob
import numpy as np


class Camera_Calibration:

    def __init__(self):
        self.images = glob.glob('data\\Q2_Image\\*.bmp')

        img = cv2.imread(self.images[0])
        self.img_size = (img.shape[1], img.shape[0])

        # set the nx, ny according the calibration chessboard pictures.
        self.nx = 8
        self.ny = 11

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...(6,5,0)
        self.objP = np.zeros((self.nx * self.ny, 3), np.float32)
        self.objP[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

    def Find_corner(self):

        for idx, image_name in enumerate(self.images):

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

        # Arrays to store object points and image points from all the images.
        objPoints = []  # 3d points in real world space
        imgPoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for idx, image_name in enumerate(self.images):

            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret:
                objPoints.append(self.objP)
                imgPoints.append(corners)

        # Do camera calibration given object points and image points
        ret, mtx, dist, rVec, tVec = cv2.calibrateCamera(objPoints, imgPoints, self.img_size, None, None)

        print("Intrinsic : ")
        print(mtx, "\n")

    def Find_Extrinsic(self, index):

        # Arrays to store object points and image points from all the images.
        objPoints = []  # 3d points in real world space
        imgPoints = []  # 2d points in image plane.

        img = cv2.imread('data\\Q2_Image\\'+index+'.bmp')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

        if ret:
            objPoints.append(self.objP)
            imgPoints.append(corners)

        # Do camera calibration given object points and image points
        ret, mtx, dist, rVec, tVec = cv2.calibrateCamera(objPoints, imgPoints, self.img_size, None, None)

        rMat, _ = cv2.Rodrigues(rVec[0])
        Extrinsic = np.concatenate((rMat, tVec[0]), axis=1)

        print("Extrinsic :")
        print(Extrinsic, "\n")

    def Find_Distortion(self):

        # Arrays to store object points and image points from all the images.
        objPoints = []  # 3d points in real world space
        imgPoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for idx, image_name in enumerate(self.images):

            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret:
                objPoints.append(self.objP)
                imgPoints.append(corners)

        # Do camera calibration given object points and image points
        ret, mtx, dist, rVec, tVec = cv2.calibrateCamera(objPoints, imgPoints, self.img_size, None, None)

        print("distortion : ")
        print(dist, "\n")

    def Show_Undistorted(self):

        # Arrays to store object points and image points from all the images.
        objPoints = []  # 3d points in real world space
        imgPoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for idx, image_name in enumerate(self.images):

            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret:
                objPoints.append(self.objP)
                imgPoints.append(corners)

        # Do camera calibration given object points and image points
        ret, mtx, dist, rVec, tVec = cv2.calibrateCamera(objPoints, imgPoints, self.img_size, None, None)
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, self.img_size, 1, self.img_size)

        for idx, image_name in enumerate(self.images):
            # undistorted
            img = cv2.imread(image_name)
            dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            dst = cv2.resize(dst, (800, 800))

            # concatenate
            image = np.concatenate((dst, cv2.resize(img, (800, 800))), axis=1)

            cv2.imshow("dst", image)
            cv2.waitKey(500)


if __name__ == '__main__':

    d = Camera_Calibration()
    d.Find_corner()
