"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
import mpl_toolkits.mplot3d as mplot3d
import itertools
import cv2
import torch


class ReprojectionTool:
    def __init__(self, root_dir, calib_paths):
        self.cameras = {}
        for camera in calib_paths:
            self.cameras[camera] = Camera(camera,
                        os.path.join(root_dir, calib_paths[camera]))
        self.camera_list = [self.cameras[cam] for cam in self.cameras]
        self.num_cameras = len(self.camera_list)
        self.cameraMatrices = torch.zeros(self.num_cameras, 4,3)
        self.intrinsicMatrices = torch.zeros(self.num_cameras, 3,3)
        self.distortionCoefficients = torch.zeros(self.num_cameras, 1,5)
        for i,cam in enumerate(self.cameras):
            self.cameraMatrices[i] =  torch.from_numpy(
                        self.cameras[cam].cameraMatrix).transpose(0,1)
            self.intrinsicMatrices[i] = torch.from_numpy(
                        self.cameras[cam].intrinsicMatrix)
            self.distortionCoefficients[i] = torch.from_numpy(
                        self.cameras[cam].distortionCoeffccients)


    def reprojectPoint(self,point3D):
        pointsRepro = np.zeros((self.num_cameras, 2))
        for i,cam in enumerate(self.camera_list):
            pointRepro = cam.cameraMatrix.dot(
                        np.concatenate((point3D, np.array([1]))))
            pointRepro = (pointRepro/pointRepro[-1])[:2]
            pointRepro[0] = ((pointRepro[0] - cam.intrinsicMatrix[2,0])
                        / cam.intrinsicMatrix[0,0])
            pointRepro[1] = ((pointRepro[1] - cam.intrinsicMatrix[2,1])
                        / cam.intrinsicMatrix[1,1])
            r2 = pointRepro[0]*pointRepro[0]+pointRepro[1]*pointRepro[1]
            pointRepro[0] = pointRepro[0] * (1+cam.distortionCoeffccients[0][0]
                        * r2 + cam.distortionCoeffccients[0][1]*r2*r2)
            pointRepro[1] = pointRepro[1] * (1+cam.distortionCoeffccients[0][0]
                        * r2 + cam.distortionCoeffccients[0][1]*r2*r2)
            pointRepro[0] = (pointRepro[0] * cam.intrinsicMatrix[0,0]
                        + cam.intrinsicMatrix[2,0])
            pointRepro[1] = (pointRepro[1] * cam.intrinsicMatrix[1,1]
                        + cam.intrinsicMatrix[2,1])
            pointsRepro[i] = pointRepro
        return pointsRepro


    def reconstructPoint(self,points, camsToUse = None):
        if camsToUse == None:
            camsToUse = range(len(self.cameras))
        if (len(camsToUse) > 1):
            camMats = []
            distCoeffs = []
            intrinsicMats = []
            for i,camera in enumerate(self.cameras):
                if i in camsToUse:
                    cam = self.cameras[camera]
                    camMats.append(cam.cameraMatrix)
                    distCoeffs.append(cam.distortionCoeffccients)
                    intrinsicMats.append(cam.intrinsicMatrix)

            pointsToUse = np.zeros((2, len(camsToUse)))
            for i,cam in enumerate(camsToUse):
                points_distorted = points[:,cam]
                camera = self.camera_list[cam]
                points_undist = cv2.undistortPoints(points_distorted,
                            camera.intrinsicMatrix.transpose(),
                            camera.distortionCoeffccients).squeeze()
                points_undist[0] = (points_undist[0]
                            * camera.intrinsicMatrix[0,0]
                            + camera.intrinsicMatrix[2,0])
                points_undist[1] = (points_undist[1]
                            * camera.intrinsicMatrix[1,1]
                            + camera.intrinsicMatrix[2,1])
                pointsToUse[:,i] = points_undist
            A = np.zeros((pointsToUse.shape[1]*2, 4))
            for i in range(pointsToUse.shape[1]):
                A[2*i:2*i+2] = pointsToUse[:, i].reshape(2,1).dot(
                            camMats[i][2].reshape(1,4)) - camMats[i][0:2]
            _,_,vh = np.linalg.svd(A)
            V = np.transpose(vh)
            X = V[:,-1]
            X = X/X[-1]
            X = X[0:3]
            return X
        else:
            return np.array([0,0,0])


class Camera:
    def __init__(self, name, calib_path):
        self.name = name
        self.position = self.get_mat_from_file(calib_path, 'T')
        self.rotationMatrix = self.get_mat_from_file(calib_path, 'R')
        self.intrinsicMatrix = self.get_mat_from_file(calib_path,
                    'intrinsicMatrix')
        self.distortionCoeffccients = self.get_mat_from_file(calib_path,
                    'distortionCoefficients')
        self.cameraMatrix = np.transpose((np.concatenate((self.rotationMatrix,
                    self.position.reshape(1,3)), axis = 0)).dot(
                    self.intrinsicMatrix))

    def get_mat_from_file(self, filepath, nodeName):
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        return fs.getNode(nodeName).mat()
