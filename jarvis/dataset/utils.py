"""
utils.py
=================
Dataset Utilities.
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
            self.cameras[camera] = Camera(camera, os.path.join(root_dir, calib_paths[camera]))
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
            pointRepro = cam.cameraMatrix.dot(np.concatenate((point3D, np.array([1]))))
            pointRepro = (pointRepro/pointRepro[-1])[:2]
            pointRepro[0] = (pointRepro[0] - cam.intrinsicMatrix[2,0]) / cam.intrinsicMatrix[0,0]
            pointRepro[1] = (pointRepro[1] - cam.intrinsicMatrix[2,1]) / cam.intrinsicMatrix[1,1]
            r2 = pointRepro[0]*pointRepro[0]+pointRepro[1]*pointRepro[1]
            pointRepro[0] = pointRepro[0]*(1+cam.distortionCoeffccients[0][0]*r2 + cam.distortionCoeffccients[0][1]*r2*r2)
            pointRepro[1] = pointRepro[1]*(1+cam.distortionCoeffccients[0][0]*r2 + cam.distortionCoeffccients[0][1]*r2*r2)
            pointRepro[0] = pointRepro[0] * cam.intrinsicMatrix[0,0] + cam.intrinsicMatrix[2,0]
            pointRepro[1] = pointRepro[1] * cam.intrinsicMatrix[1,1] + cam.intrinsicMatrix[2,1];
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
                points_undist = cv2.undistortPoints(points_distorted, camera.intrinsicMatrix.transpose(),camera.distortionCoeffccients).squeeze()
                points_undist[0] = points_undist[0]*camera.intrinsicMatrix[0,0]+camera.intrinsicMatrix[2,0]
                points_undist[1] = points_undist[1]*camera.intrinsicMatrix[1,1]+camera.intrinsicMatrix[2,1]
                pointsToUse[:,i] = points_undist
            A = np.zeros((pointsToUse.shape[1]*2, 4))
            for i in range(pointsToUse.shape[1]):
                A[2*i:2*i+2] =  pointsToUse[:, i].reshape(2,1).dot(camMats[i][2].reshape(1,4)) - camMats[i][0:2]
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
        self.intrinsicMatrix = self.get_mat_from_file(calib_path, 'intrinsicMatrix')
        self.distortionCoeffccients = self.get_mat_from_file(calib_path, 'distortionCoefficients')
        self.cameraMatrix = np.transpose((np.concatenate((self.rotationMatrix, self.position.reshape(1,3)), axis = 0)).dot(self.intrinsicMatrix))

    def get_mat_from_file(self, filepath, nodeName):
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        return fs.getNode(nodeName).mat()


# class SetupVisualizer():
#     def __init__(self, primary_camera, root_dir, intrinsics, extrinsics):
#         self.cameras = {}
#         self.root_dir = root_dir
#         for camera in intrinsics:
#             if camera == primary_camera:
#                 self.cameras[camera] = Camera(camera, True,
#                             os.path.join(root_dir, intrinsics[camera]), None)
#             else:
#                 self.cameras[camera] = Camera(camera, False,
#                             os.path.join(root_dir, intrinsics[camera]),
#                             os.path.join(root_dir, extrinsics[camera]))
#         self.camera_list = [self.cameras[cam] for cam in self.cameras]
#         self.num_cameras = len(self.camera_list)
#
#
#     def plot_cameras(self, axes = None):
#         show_plot = False
#         if axes == None:
#             figure = plt.figure()
#             axes = figure.gca(projection='3d')
#             show_plot = True
#
#         cam_positions = []
#         for cam in self.cameras:
#             current_dir = os.path.dirname(
#                         os.path.abspath(inspect.getfile(
#                         inspect.currentframe())))
#             camera_mesh = mesh.Mesh.from_file(
#                         os.path.join(current_dir, 'Camera.stl'))
#             camera_mesh.points = camera_mesh.points*0.7
#             rotMat = np.eye(3)
#             rotMat[0,0] = -1
#             rotMat[1,1] = 1
#             rotMat[2,2] = -1
#             cam_positions.append(np.dot(rotMat,np.dot(
#                         self.cameras[cam].rotationMatrix,
#                         self.cameras[cam].position)))
#             camera_mesh.rotate_using_matrix(rotMat)
#             camera_mesh.translate(-self.cameras[cam].position)
#             camera_mesh.rotate_using_matrix(
#                         np.transpose(self.cameras[cam].rotationMatrix))
#             if self.cameras[cam].primary:
#                 axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
#                             camera_mesh.vectors, color='r'))
#             else:
#                 axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
#                             camera_mesh.vectors, color='b'))
#
#         cam_positions = np.array(cam_positions).squeeze()
#         x_range = np.array([np.min(cam_positions[:,0]),
#                             np.max(cam_positions[:,0])])
#         y_range = np.array([np.min(cam_positions[:,1]),
#                             np.max(cam_positions[:,1])])
#         z_range = np.array([np.min(cam_positions[:,2]),
#                             np.max(cam_positions[:,2])])
#         ranges = np.array([x_range, y_range, z_range])
#         max_range = np.max(ranges[:,1]-ranges[:,0])
#         centers = (ranges[:,1]+ranges[:,0])/2.
#
#         axes.set_xlim3d(centers[0]-max_range/2, centers[0]+max_range/2)
#         axes.set_ylim3d(centers[1]-max_range/2, centers[1]+max_range/2)
#         axes.set_zlim3d(centers[2]-max_range/2, centers[2]+max_range/2)
#         axes.set_xlabel('X')
#         axes.set_ylabel('Y')
#         axes.set_zlabel('Z')
#         plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
#
#         if show_plot:
#             plt.show()
#
#
#     def plot_tracking_area(self, tracking_area, axes = None):
#         show_plot = False
#         if axes == None:
#             figure = plt.figure()
#             axes = figure.gca(projection='3d')
#             show_plot = True
#         tracking_area = tracking_area.transpose()
#         cube = np.array(list(itertools.product(*zip(tracking_area[0],
#                     tracking_area[1]))))
#         cube = cube
#         for corner in cube:
#             axes.scatter(corner[0],corner[1],corner[2], c = 'green', s = 20)
#         vertices = [[0,1], [0,2], [2,3], [1,3], [4,5], [4,6], [6,7], [5,7],
#                     [0,4], [1,5], [2,6], [3,7]]
#         for vertex in vertices:
#             axes.plot([cube[vertex[0]][0],cube[vertex[1]][0]],
#                       [cube[vertex[0]][1],cube[vertex[1]][1]],
#                       [cube[vertex[0]][2],cube[vertex[1]][2]], c = 'green')
#         if show_plot:
#             plt.show()
#
#
#     def plot_datapoints(self, points, axes):
#         show_plot = False
#         if axes == None:
#             figure = plt.figure()
#             axes = figure.gca(projection='3d')
#             show_plot = True
#         for i, point in enumerate(points):
#             axes.scatter(point[0], point[1], point[2])
#
#         if show_plot:
#             plt.show()
