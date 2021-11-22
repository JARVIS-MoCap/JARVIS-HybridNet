import os,sys,inspect
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolorsf


class ReprojectionLayer(nn.Module):
    def __init__(self, cfg, num_cameras = None):
        super(ReprojectionLayer, self).__init__()
        self.cfg = cfg
        dataset_dir = os.path.join(cfg.DATASET.DATASET_ROOT_DIR,
                                   cfg.DATASET.DATASET_3D)

        self.register_buffer('grid_spacing',
                    torch.tensor(self.cfg.HYBRIDNET.GRID_SPACING))
        self.register_buffer('boxsize',
                    torch.tensor(self.cfg.HYBRIDNET.ROI_CUBE_SIZE))
        self.register_buffer('grid_size',
                    torch.tensor(self.cfg.HYBRIDNET.ROI_CUBE_SIZE
                    / self.cfg.HYBRIDNET.GRID_SPACING).int())
        if num_cameras:
            self.register_buffer('num_cameras',
                        torch.tensor(num_cameras))
        else:
            self.register_buffer('num_cameras',
                        torch.tensor(self.cfg.DATASET.NUM_CAMERAS))
        self.register_buffer('grid_size',
                    torch.tensor(self.cfg.HYBRIDNET.ROI_CUBE_SIZE
                    / self.cfg.HYBRIDNET.GRID_SPACING).int())
        self.register_buffer('img_size',
                    torch.tensor(cfg.DATASET.IMAGE_SIZE))

        self.ii,self.xx,self.yy,self.zz = torch.meshgrid(
                    torch.arange(self.num_cameras).cuda(),
                    torch.arange(self.grid_size).cuda(),
                    torch.arange(self.grid_size).cuda(),
                    torch.arange(self.grid_size).cuda())

        self.grid = torch.zeros((self.grid_size, self.grid_size,
                                 self.grid_size,3))
        half_gridsize = int(self.grid_size/2)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[i,j,k] = torch.tensor([i - half_gridsize,
                                                     j - half_gridsize,
                                                     k - half_gridsize])
        self.grid = self.grid.cuda()
        self.grid = self.grid * self.grid_spacing


    def reprojectPoints(self, x, cameraMatrices):
      res = torch.cuda.LongTensor(self.num_cameras, self.grid_size,
                                  self.grid_size, self.grid_size)
      ones = torch.ones([x.shape[0], x.shape[1], x.shape[2],1]).cuda()
      x = torch.cat((x,ones),3)
      for i in range(self.num_cameras):
          partial = torch.matmul(x, cameraMatrices[i])
          partial[:,:,:,0] = torch.clamp(partial[:,:,:,0]
                    / partial[:,:,:,2], 0, self.img_size[0] - 1)
          partial[:,:,:,1] = torch.clamp(partial[:,:,:,1]
                    / partial[:,:,:,2], 0, self.img_size[1] - 1)
          res[i] = ((partial[:,:,:,1] / 2).int() * (self.img_size[0] / 2).int()
                    + (partial[:,:,:,0] / 2).int())
      return res


    def _get_heatmap_value(self, heatmaps, grid, cameraMatrices):
        heatmaps = heatmaps.flatten(2)
        reproPoints = self.reprojectPoints(grid,cameraMatrices)
        outs = torch.mean(
            heatmaps[:,self.ii,reproPoints[self.ii,self.xx,self.yy,self.zz]],
            dim = 1)
        return outs


    def forward(self, heatmaps, center, cameraMatrices):
        heatmaps3D = torch.cuda.FloatTensor(heatmaps.shape[0],
                    heatmaps.shape[2], self.grid_size, self.grid_size,
                    self.grid_size)
        for batch in range(heatmaps.shape[0]):
            grid = self.grid+center[batch]
            heatmaps3D[batch] = self._get_heatmap_value(torch.transpose(
                        heatmaps[batch], 0,1), grid, cameraMatrices[batch])
        return heatmaps3D
