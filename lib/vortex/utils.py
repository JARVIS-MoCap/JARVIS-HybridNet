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
        for camera in intrinsics:
            if camera == primary_camera:
                self.cameras[camera] = Camera(camera, True, os.path.join(root_dir, intrinsics[camera]), None)
            else:
                self.cameras[camera] = Camera(camera, False, os.path.join(root_dir, intrinsics[camera]), os.path.join(root_dir, extrinsics[camera]))
        self.camera_list = [self.cameras[cam] for cam in self.cameras]
        self.num_cameras = len(self.camera_list)

    def reprojectPoint(self,point3D):
        pointsRepro = np.zeros((self.num_cameras, 2))
        for i,cam in enumerate(self.camera_list):
            pointRepro = cam.cameraMatrix.dot(np.concatenate((point3D, np.array([1]))))
            pointRepro = (pointRepro/pointRepro[-1])[:2]
            pointRepro[0] = max(0, min(pointRepro[0],1279))
            pointRepro[1] = max(0, min(pointRepro[1],1023))
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

    def renderPoints(self, points, show_cameras = True, axes = None):
        # Create a new plot
        if axes == None:
            figure = plt.figure()
            axes = figure.gca(projection='3d')

        if show_cameras:
            for cam in self.cameras:
                camera_mesh = mesh.Mesh.from_file('/home/lambda/Documents/TrackingDMDS/CombiNet/Sony RX1Rii_3dless_com_simplified.stl')
                camera_mesh.points = camera_mesh.points*0.7
                rotMat = np.eye(3)
                rotMat[0,0] = -1
                rotMat[1,1] = 1
                rotMat[2,2] = -1
                camera_mesh.rotate_using_matrix(rotMat)
                camera_mesh.translate(-self.cameras[cam].position)
                camera_mesh.rotate_using_matrix(np.transpose(self.cameras[cam].rotationMatrix))
                if self.cameras[cam].primary:
                    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(camera_mesh.vectors, color='r'))
                else:
                    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(camera_mesh.vectors, color='b'))

        c = ['r', 'r','r','r','b','b','b','b','g','g','g','g', 'orange', 'orange','orange','orange', 'y','y','y','y','grey', 'grey','grey']
        for i, point in enumerate(points):
            #axes.scatter(point[0], point[1], point[2], color = c[i])
            print ("Classic:", i, point)

        axes.set_xlim3d(-600, 600)
        axes.set_ylim3d(600, -600)
        axes.set_zlim3d(1200, 0)
        axes.set_xlabel('X Label')
        axes.set_ylabel('Y Label')
        axes.set_zlabel('Z Label')
        plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)

        if axes == None:
            plt.show()

class Camera:
    def __init__(self, name, primary, intrinsics, extrinsics = None):
        self.name = name
        self.primary = primary
        if self.primary:
            self.position = np.array([0.,0.,0.])
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
