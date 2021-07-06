"""
utils.py
===============
"""

import math
import os
from glob import glob
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
import torch
from torch import nn


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus, mode):
        super().__init__(module)
        self.num_gpus = num_gpus
        self.mode = mode

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')
        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[2][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[3][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[4][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)


class ReprojectionTool:
    def __init__(self, primary_camera, root_dir, intrinsics, extrinsics):
        self.cameras = {}
        found_primary = False
        for camera in intrinsics:
            if camera == primary_camera:
                self.cameras[camera] = Camera(camera, True, os.path.join(root_dir, intrinsics[camera]), None)
                found_primary = True
            else:
                self.cameras[camera] = Camera(camera, False, os.path.join(root_dir, intrinsics[camera]), os.path.join(root_dir, extrinsics[camera]))
        assert found_primary, 'Primary camera name does not match any of the given cameras'

        self.camera_list = [self.cameras[cam] for cam in self.cameras]
        self.num_cameras = len(self.camera_list)
        resolutions = [self.cameras[cam].resolution for cam in self.cameras]
        assert resolutions[:-1] == resolutions[1:], 'All cameras need to record at the same resolution'
        self.resolution = resolutions[0]

    def reprojectPoint(self,point3D):
        pointsRepro = np.zeros((self.num_cameras, 2))
        for i,cam in enumerate(self.camera_list):
            pointRepro = cam.cameraMatrix.dot(np.concatenate((point3D, np.array([1]))))
            pointRepro = (pointRepro/pointRepro[-1])[:2]
            pointRepro[0] = max(0, min(pointRepro[0],cam.resolution[0]-1))
            pointRepro[1] = max(0, min(pointRepro[1],cam.resolution[1]-1))
            pointsRepro[i] = pointRepro
        return pointsRepro

    def reconstructPoint(self,points):
        camMats = []
        for camera in self.cameras:
            cam = self.cameras[camera]
            camMats.append(cam.cameraMatrix)
        A = np.zeros((points.shape[1]*2, 4))
        for i in range(points.shape[1]):
            A[2*i:2*i+2] =  points[:, i].reshape(2,1).dot(camMats[i][2].reshape(1,4)) - camMats[i][0:2]
        _,_,vh = np.linalg.svd(A)
        V = np.transpose(vh)
        X = V[:,-1]
        X = X/X[-1]
        X = X[0:3]
        return X


class Camera:
    def __init__(self, name, primary, intrinsics, extrinsics = None):
        self.name = name
        self.primary = primary
        self.resolution = [1280,1024] #TODO: Include this in intrinsics file
        if self.primary:
            self.position = np.array([0.,0.,0.]).reshape(3,1)
            self.rotationMatrix = np.eye(3)
        else:
            self.position = self.get_mat_from_file(extrinsics, 'T')
            self.rotationMatrix = self.get_mat_from_file(extrinsics, 'R')
        self.intrinsicMatrix = self.get_mat_from_file(intrinsics, 'intrinsicMatrix')
        self.distortionCoeffccients = self.get_mat_from_file(intrinsics, 'distortionCoeffccients')
        self.cameraMatrix = np.transpose((np.concatenate((self.rotationMatrix, self.position.reshape(1,3)), axis = 0)).dot(self.intrinsicMatrix))

    def get_mat_from_file(self, filepath, nodeName):
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        return fs.getNode(nodeName).mat()
