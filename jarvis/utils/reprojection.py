"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import os
import json
import cv2

import torch
import torch.nn as nn


class ReprojectionTool(nn.Module):
    def __init__(self, root_dir = None, calib_paths = None, device = 'cuda'):
        super(ReprojectionTool, self).__init__()
        self.device = device
        if calib_paths != None:
            self.cameras = {}
            for camera in calib_paths:
                self.cameras[camera] = TorchCamera(camera,
                            os.path.join(root_dir, calib_paths[camera]), device)
            self.camera_list = [self.cameras[cam] for cam in self.cameras]
            self.num_cameras = len(self.camera_list)
            self.cameraMatrices = torch.zeros((self.num_cameras, 4,3),
                        device = torch.device(device))
            self.intrinsicMatrices = torch.zeros((self.num_cameras, 3,3),
                        device = torch.device(device))
            self.distortionCoefficients = torch.zeros((self.num_cameras, 1,5),
                        device = torch.device(device))
            for i,cam in enumerate(self.cameras):
                self.cameraMatrices[i] = \
                            self.cameras[cam].cameraMatrix.transpose(0,1)
                self.intrinsicMatrices[i] = \
                            self.cameras[cam].intrinsicMatrix
                self.distortionCoefficients[i] = \
                            self.cameras[cam].distortionCoeffccients
        else:
            self.cameraMatrices = torch.tensor(0,
                        device = torch.device(device))
            self.intrinsicMatrices = torch.tensor(0,
                        device = torch.device(device))
            self.distortionCoefficients = torch.tensor(0,
                        device = torch.device(device))


    def reprojectPoint(self,point3D):
        ones = torch.ones([point3D.shape[0],1],
                    device=torch.device(self.device))
        point3D = torch.cat((point3D, ones),1).unsqueeze(0)
        pointRepro = torch.matmul(point3D, self.cameraMatrices).permute(1,2,0)
        pointRepro[:,0] = (pointRepro[:,0] / pointRepro[:,2]
                    - self.intrinsicMatrices[:,2,0])
        pointRepro[:,1] = (pointRepro[:,1] / pointRepro[:,2]
                    - self.intrinsicMatrices[:,2,1])
        r2 = (torch.square(pointRepro[:,0] / self.intrinsicMatrices[:,0,0])
                    + torch.square(pointRepro[:,1]
                    / self.intrinsicMatrices[:,1,1]))
        distort = (1 + (self.distortionCoefficients[:, 0, 0]
                    + self.distortionCoefficients[: ,0, 1]*r2)*r2)
        pointRepro[:,0] = pointRepro[:,0]*distort+self.intrinsicMatrices[:,2,0]
        pointRepro[:,1] = pointRepro[:,1]*distort+self.intrinsicMatrices[:,2,1]
        pointRepro = pointRepro[:,:2].permute(0,2,1).squeeze()
        return pointRepro


    def reconstructPoint(self,points, maxvals):
        cameraMatrices = self.cameraMatrices.permute(0,2,1)
        points[0] = (points[0]-self.intrinsicMatrices[:,2,0])
        points[1] = (points[1]-self.intrinsicMatrices[:,2,1])
        r2 = (torch.square(points[0] / self.intrinsicMatrices[:,0,0])
                    + torch.square(points[1] / self.intrinsicMatrices[:,1,1]))
        distort = (1+(self.distortionCoefficients[:, 0, 0]
                    + self.distortionCoefficients[: ,0, 1]*r2)*r2)
        points[0] = points[0]/distort+self.intrinsicMatrices[:,2,0]
        points[1] = points[1]/distort+self.intrinsicMatrices[:,2,1]

        A = (torch.bmm(points.permute(1,0).reshape(points.shape[1], 2, 1),
                    cameraMatrices[:,2].reshape(cameraMatrices.shape[0], 1, 4))
                    - cameraMatrices[:,0:2])
        A = A * maxvals

        _,_,vh = torch.linalg.svd(A.flatten(0,1))
        V = vh.transpose(0,1)
        X = V[:,-1]
        X = X/X[-1]
        X = X[0:3]
        return X


class TorchCamera(nn.Module):
    def __init__(self, name, calib_path, device = 'cuda'):
        super(TorchCamera, self).__init__()
        self.name = name
        self.position = torch.from_numpy(self.get_mat_from_file(
                    calib_path, 'T')).float().to(device)
        self.rotationMatrix = torch.from_numpy(self.get_mat_from_file(
                    calib_path, 'R')).float().to(device)
        self.intrinsicMatrix = torch.from_numpy(self.get_mat_from_file(
                    calib_path, 'intrinsicMatrix')).float().to(device)
        self.distortionCoeffccients = torch.from_numpy(self.get_mat_from_file(
                    calib_path, 'distortionCoefficients')).float().to(device)
        self.cameraMatrix = torch.transpose(torch.matmul(
                    torch.cat((self.rotationMatrix, self.position.reshape(1,3)),
                    axis = 0), (self.intrinsicMatrix)), 0,1).float().to(device)

    def get_mat_from_file(self, filepath, nodeName):
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        return fs.getNode(nodeName).mat()



def get_repro_tool(cfg, dataset_name, device = 'cuda'):
    reproTools = load_reprojection_tools(cfg, device = device)
    if dataset_name != None and not dataset_name in reproTools:
        if os.path.isdir(dataset_name):
            dataset_dir = os.path.join(cfg.PARENT_DIR,
                        cfg.DATASET.DATASET_ROOT_DIR,
                        cfg.DATASET.DATASET_3D)
            dataset_json = open(os.path.join(dataset_dir, 'annotations',
                        'instances_val.json'))
            data = json.load(dataset_json)
            calibPaths = {}
            calibParams = list(data['calibrations'].keys())[0]
            for cam in data['calibrations'][calibParams]:
                calibPaths[cam] = \
                        data['calibrations'][calibParams][cam].split("/")[-1]
            reproTool = ReprojectionTool(dataset_name, calibPaths, device)
        else:
            print (f'{CLIColors.FAIL}Could not load reprojection Tool for'
                        f'specified project...{CLIColors.ENDC}')
            return None
    elif len(reproTools) == 1:
        reproTool = reproTools[list(reproTools.keys())[0]]
    elif len(reproTools) > 1:
        if dataset_name == None:
            reproTool = reproTools[list(reproTools.keys())[0]]
        else:
            reproTool = reproTools[dataset_name]
    else:
        print (f'{CLIColors.FAIL}Could not load reprojection Tool for specified'
                    f' project...{CLIColors.ENDC}')
        return None
    return reproTool


def load_reprojection_tools(cfg, cameras_to_use = None, device = 'cuda'):
    if cameras_to_use != None:
        print (f"Using subset of cameras: {cameras_to_use}.")
    dataset_dir = os.path.join(cfg.PARENT_DIR, cfg.DATASET.DATASET_ROOT_DIR,
                cfg.DATASET.DATASET_3D)
    dataset_json = open(os.path.join(dataset_dir, 'annotations',
                'instances_val.json'))
    data = json.load(dataset_json)
    reproTools = {}
    for calibParams in data['calibrations']:
        calibPaths = {}
        for cam in data['calibrations'][calibParams]:
            if cameras_to_use == None or cam in cameras_to_use:
                calibPaths[cam] = data['calibrations'][calibParams][cam]
        reproTools[calibParams] = ReprojectionTool(
                    dataset_dir,calibPaths, device)
    dataset_json.close()
    return reproTools
