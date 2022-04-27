"""
model.py
========
HybridNet torch module.
"""

import os,sys,inspect
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from .repro_layer import ReprojectionLayer
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.efficienttrack.model import EfficientTrackBackbone
import jarvis.efficienttrack.darkpose as darkpose
from .v2vnet import V2VNet
import time


class HybridNetBackbone(nn.Module):
    def __init__(self, cfg, efficienttrack_weights = None):
        super(HybridNetBackbone, self).__init__()
        self.cfg = cfg
        self.root_dir = cfg.DATASET.DATASET_ROOT_DIR
        self.grid_spacing = torch.tensor(cfg.HYBRIDNET.GRID_SPACING)
        self.grid_size = torch.tensor(cfg.HYBRIDNET.ROI_CUBE_SIZE)

        self.effTrack = EfficientTrackBackbone(self.cfg.KEYPOINTDETECT,
                    model_size=self.cfg.KEYPOINTDETECT.MODEL_SIZE,
                    output_channels = self.cfg.KEYPOINTDETECT.NUM_JOINTS)

        if efficienttrack_weights != None:
            self.effTrack.load_state_dict(torch.load(efficienttrack_weights),
                        strict = True)

        self.reproLayer = ReprojectionLayer(cfg)
        self.v2vNet = V2VNet(cfg.KEYPOINTDETECT.NUM_JOINTS,
                             cfg.KEYPOINTDETECT.NUM_JOINTS)

        self.softplus = nn.Softplus()
        self.xx,self.yy,self.zz = torch.meshgrid(
                torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda(),
                torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda(),
                torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda(),
                indexing = 'ij')
        self.last_time = 0


        self.heatmap_size = torch.cuda.IntTensor([0,0])


    def forward(self, imgs, img_size, centerHM, center3D,cameraMatrices,
                intrinsicMatrices, distortionCoefficients):
        batch_size = imgs.shape[0]
        self.heatmap_size = (img_size/2).int()
        heatmaps_batch = self.effTrack(
                imgs.reshape(-1,imgs.shape[2], imgs.shape[3], imgs.shape[4]))[1]

        heatmaps_batch = heatmaps_batch.reshape(batch_size, -1,
                heatmaps_batch.shape[1],
                heatmaps_batch.shape[2],
                heatmaps_batch.shape[3])



        heatmaps_padded = F.pad(input=heatmaps_batch,
             pad = [1,1,1,1], mode='constant', value=0.)
        heatmaps3D = self.reproLayer(heatmaps_padded, center3D,centerHM,
                    cameraMatrices, intrinsicMatrices, distortionCoefficients)
        if (self.training):
            heatmaps3D = self.drop_joint(heatmaps3D)    

        heatmap_final = self.v2vNet((heatmaps3D/255.))
        heatmap_final = self.softplus(heatmap_final)
        #heatmap_final = heatmaps_gt
        #TODO: Make this work for different batch sizes"!!
        norm = torch.sum(heatmap_final, dim = [2,3,4])
        x = torch.mul(heatmap_final, self.xx)
        x = torch.sum(x, dim = [2,3,4])/norm
        y = torch.mul(heatmap_final, self.yy)
        y = torch.sum(y, dim = [2,3,4])/norm
        z = torch.mul(heatmap_final, self.zz)
        z = torch.sum(z, dim = [2,3,4])/norm
        points3D = torch.stack([x,y,z], dim = 2)
        points3D = (points3D.transpose(0,1)*self.grid_spacing*2 - self.grid_size
                    / 2. + center3D).transpose(0,1)
        heatmap_final = self.softplus(heatmap_final)

        return heatmap_final, heatmaps_padded, points3D
