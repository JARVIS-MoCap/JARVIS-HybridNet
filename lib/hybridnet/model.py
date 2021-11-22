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

from lib.hybridnet.repro_layer import ReprojectionLayer
from lib.dataset.dataset3D import Dataset3D
from lib.hybridnet.efficienttrack.model import EfficientTrackBackbone
import lib.hybridnet.efficienttrack.darkpose as darkpose
from lib.hybridnet.v2vnet import V2VNet

# self.starter, self.ender = torch.cuda.Event(enable_timing=True),
#         torch.cuda.Event(enable_timing=True)
# self.starter.record()
# self.ender.record()


class HybridNetBackbone(nn.Module):
    def __init__(self, cfg, efficienttrack_weights = None):
        super(HybridNetBackbone, self).__init__()
        self.cfg = cfg
        self.root_dir = cfg.DATASET.DATASET_ROOT_DIR
        self.register_buffer('grid_size',
                    torch.tensor(cfg.HYBRIDNET.ROI_CUBE_SIZE))
        self.register_buffer('grid_spacing',
                    torch.tensor(cfg.HYBRIDNET.GRID_SPACING))

        self.effTrack = EfficientTrackBackbone(self.cfg.KEYPOINTDETECT,
                    compound_coef=self.cfg.KEYPOINTDETECT.COMPOUND_COEF,
                    output_channels = self.cfg.KEYPOINTDETECT.NUM_JOINTS)
        if efficienttrack_weights != None:
            self.effTrack.load_state_dict(torch.load(efficienttrack_weights),
                        strict = True)

        self.reproLayer = ReprojectionLayer(cfg)
        img_size = self.cfg.DATASET.IMAGE_SIZE
        self.register_buffer('heatmap_size', (torch.tensor([int(img_size[0]/2),
                    int(img_size[1]/2)])))
        self.v2vNet = V2VNet(cfg.KEYPOINTDETECT.NUM_JOINTS,
                             cfg.KEYPOINTDETECT.NUM_JOINTS)
        self.softplus = nn.Softplus()
        self.xx,self.yy,self.zz = torch.meshgrid(
                torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda(),
                torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda(),
                torch.arange(int(self.grid_size/self.grid_spacing/2)).cuda())
        self.last_time = 0


    def forward(self, imgs, centerHM, center3D, cameraMatrices):
        batch_size = imgs.shape[0]
        heatmaps_batch = self.effTrack(
                imgs.reshape(-1,imgs.shape[2], imgs.shape[3], imgs.shape[4]))[1]

        heatmaps_batch = heatmaps_batch.reshape(batch_size, -1,
                heatmaps_batch.shape[1],
                heatmaps_batch.shape[2],
                heatmaps_batch.shape[3])

        heatmaps_padded = torch.cuda.FloatTensor(
                imgs.shape[0], imgs.shape[1], heatmaps_batch.shape[2],
                self.heatmap_size[1], self.heatmap_size[0])
        heatmaps_padded.fill_(0)

        for i in range(imgs.shape[1]):
            heatmaps = heatmaps_batch[:,i]
            for batch, heatmap in enumerate(heatmaps):
                heatmaps_padded[batch,i] = F.pad(input=heatmap,
                     pad = (((centerHM[batch,i,0]/2)-heatmap.shape[-1]/2).int(),
                          self.heatmap_size[0]-((centerHM[batch,i,0]/2)
                          + heatmap.shape[-1]/2).int(),
                          ((centerHM[batch,i,1]/2)-heatmap.shape[-1]/2).int(),
                          self.heatmap_size[1]-((centerHM[batch,i,1]/2)
                          + heatmap.shape[-1]/2).int()),
                     mode='constant', value=0)

        heatmaps3D = self.reproLayer(heatmaps_padded, center3D, cameraMatrices)
        heatmap_final = self.v2vNet(((heatmaps3D/255.)))
        heatmap_final = self.softplus(heatmap_final)
        #TODO: Make this work for different batch sizes"!!
        norm = torch.sum(heatmap_final, dim = [2,3,4])
        x = torch.mul(heatmap_final, self.xx)
        x = torch.sum(x, dim = [2,3,4])/norm
        y = torch.mul(heatmap_final, self.yy)
        y = torch.sum(y, dim = [2,3,4])/norm
        z = torch.mul(heatmap_final, self.zz)
        z = torch.sum(z, dim = [2,3,4])/norm
        points3D = torch.stack([x,y,z], dim = 2)
        points3D = (points3D*self.grid_spacing*2 - self.grid_size
                    / self.grid_spacing + center3D)
        return heatmap_final, heatmaps_padded, points3D
