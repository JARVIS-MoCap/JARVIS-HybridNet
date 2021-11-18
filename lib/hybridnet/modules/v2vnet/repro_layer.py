import os,sys,inspect
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lib.hybridnet.utils import ReprojectionTool

class ReprojectionLayer(nn.Module):
    def __init__(self, cfg, intrinsic_paths, extrinsic_paths):
        super(ReprojectionLayer, self).__init__()
        self.cfg = cfg
        dataset_dir = os.path.join(cfg.DATASET.DATASET_ROOT_DIR, cfg.DATASET.DATASET_3D)
        self.reproTool = ReprojectionTool('Camera_T', dataset_dir, intrinsic_paths, extrinsic_paths)

        self.register_buffer('grid_spacing', torch.tensor(self.cfg.VORTEX.GRID_SPACING))

        self.register_buffer('boxsize', torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE))
        self.register_buffer('grid_size', torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE/self.cfg.VORTEX.GRID_SPACING).int())
        self.register_buffer('num_cameras',  torch.tensor(self.reproTool.num_cameras))
        self.register_buffer('grid_size',  torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE/self.cfg.VORTEX.GRID_SPACING).int())

        self.ii,self.xx,self.yy,self.zz = torch.meshgrid(torch.arange(self.num_cameras).cuda(),
                                                         torch.arange(self.grid_size).cuda(),
                                                         torch.arange(self.grid_size).cuda(),
                                                         torch.arange(self.grid_size).cuda())

        self.grid = torch.zeros((self.grid_size, self.grid_size, self.grid_size,3))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[i,j,k] = torch.tensor([i-int(self.grid_size/2),j-int(self.grid_size/2),k-int(self.grid_size/2)])
        self.grid = self.grid.cuda()
        self.grid = self.grid * self.grid_spacing

        self.cameraMatrices = torch.zeros(self.num_cameras, 4,3)
        for i,cam in enumerate(self.reproTool.cameras):
            self.cameraMatrices[i] =  torch.from_numpy(self.reproTool.cameras[cam].cameraMatrix).transpose(0,1)
        self.cameraMatrices = self.cameraMatrices.cuda()

        #self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


    def reprojectPoints(self, x):
      res = torch.cuda.LongTensor(12, self.grid_size, self.grid_size, self.grid_size)
      ones = torch.ones([x.shape[0], x.shape[1], x.shape[2],1]).cuda()
      x = torch.cat((x,ones),3)
      for i in range(12):
          partial = torch.matmul(x, self.cameraMatrices[i])
          partial[:,:,:,0] = torch.clamp(partial[:,:,:,0]/partial[:,:,:,2],0,1279)
          partial[:,:,:,1] = torch.clamp(partial[:,:,:,1]/partial[:,:,:,2],0,1023)
          res[i] = (partial[:,:,:,1]/2).int()*640+(partial[:,:,:,0]/2).int()
      return res


    def _get_heatmap_value(self, heatmaps, grid):
        heatmaps = heatmaps.flatten(2)
        reproPoints = self.reprojectPoints(grid)
        outs = torch.mean(heatmaps[:,self.ii,reproPoints[self.ii,self.xx,self.yy,self.zz]], dim = 1)
        return outs

    def forward(self, heatmaps, center):
        heatmaps3D = torch.cuda.FloatTensor(heatmaps.shape[0], heatmaps.shape[2], self.grid_size, self.grid_size, self.grid_size)
        for batch in range(heatmaps.shape[0]):
            grid = self.grid+center[batch]
            heatmaps3D[batch] = self._get_heatmap_value(torch.transpose(heatmaps[batch], 0,1), grid)
        return heatmaps3D
