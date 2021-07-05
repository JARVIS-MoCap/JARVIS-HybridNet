import os,sys,inspect
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lib.vortex.utils import ReprojectionTool

class ReprojectionLayer(nn.Module):
    def __init__(self, cfg, intrinsic_paths, extrinsic_paths):
        super(ReprojectionLayer, self).__init__()
        self.cfg = cfg
        lookup_path = os.path.join(cfg.PROJECTS_ROOT_PATH, cfg.PROJECT_NAME, 'lookup.npy')
        dataset_dir = os.path.join(cfg.DATASET.DATASET_ROOT_DIR, cfg.DATASET.DATASET_3D)
        self.reproTool = ReprojectionTool('T', dataset_dir, intrinsic_paths, extrinsic_paths)

        if not os.path.isfile(lookup_path):
            self.register_buffer('reproLookup', torch.from_numpy(self._create_lookup(lookup_path)).permute(3,0,1,2))
        else:
            self.reproLookup = torch.from_numpy(np.load(lookup_path).astype('int32')).permute(3,0,1,2).int()

        self.register_buffer('offset', torch.tensor([self.cfg.VORTEX.GRID_DIM_X[0], self.cfg.VORTEX.GRID_DIM_Y[0], self.cfg.VORTEX.GRID_DIM_Z[0]]))
        self.register_buffer('grid_spacing', torch.tensor(self.cfg.VORTEX.GRID_SPACING))

        self.register_buffer('boxsize', torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE))
        self.register_buffer('grid_size', torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE/self.cfg.VORTEX.GRID_SPACING).int())
        self.register_buffer('num_cameras',  torch.tensor(self.reproTool.num_cameras))
        self.register_buffer('grid_size',  torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE/self.cfg.VORTEX.GRID_SPACING).int())
        self.register_buffer('grid_size_half',  torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE/self.cfg.VORTEX.GRID_SPACING/2).int())

        self.ii,self.xx,self.yy,self.zz = torch.meshgrid(torch.arange(self.num_cameras).cuda(),
                                                         torch.arange(self.grid_size).cuda(),
                                                         torch.arange(self.grid_size).cuda(),
                                                         torch.arange(self.grid_size).cuda())
        #self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def _create_lookup(self, lookup_path):
        x = np.arange(self.cfg.VORTEX.GRID_DIM_X[0], self.cfg.VORTEX.GRID_DIM_X[1], self.cfg.VORTEX.GRID_SPACING)
        y = np.arange(self.cfg.VORTEX.GRID_DIM_Y[0], self.cfg.VORTEX.GRID_DIM_Y[1], self.cfg.VORTEX.GRID_SPACING)
        z = np.arange(self.cfg.VORTEX.GRID_DIM_Z[0], self.cfg.VORTEX.GRID_DIM_Z[1], self.cfg.VORTEX.GRID_SPACING)
        reproLookup = np.zeros((len(x),len(y),len(z), self.reproTool.num_cameras), dtype = np.int32)

        def parallel_repro(i):
            lookup = np.zeros((len(y),len(z), self.reproTool.num_cameras), dtype = np.int32)
            resolution = int(self.reproTool.resolution[0]/2)
            print (i)
            for j in range(len(y)):
                for k in range(len(z)):
                    point = self.reproTool.reprojectPoint([x[i],y[j],z[k]])
                    lookup[j,k] = (point[:,1]/2).astype(np.int32)*resolution+(point[:,0]/2).astype(np.int32)
            return i, lookup

        result = Parallel(n_jobs=-1)(delayed(parallel_repro)(i) for i in range(len(x)))
        for element in result:
            reproLookup[element[0]] = element[1]
        np.save(lookup_path, reproLookup)
        return reproLookup


    def _get_heatmap_value(self, lookup, heatmaps):
        lookup = lookup.cuda().long()
        heatmaps = heatmaps.flatten(2)
        outs = torch.mean(heatmaps[:,self.ii,lookup[self.ii,self.xx,self.yy,self.zz]], dim = 1)
        return outs

    def forward(self, heatmaps, center):
        center_indices = ((center-self.offset)/self.grid_spacing).int()
        heatmaps3D = torch.cuda.FloatTensor(heatmaps.shape[0], heatmaps.shape[2], self.grid_size, self.grid_size, self.grid_size)
        for batch in range(heatmaps.shape[0]):
            lookup_subset = self.reproLookup[:,center_indices[batch][0]-self.grid_size_half:center_indices[batch][0]+self.grid_size_half,
                                             center_indices[batch][1]-self.grid_size_half:center_indices[batch][1]+self.grid_size_half,
                                             center_indices[batch][2]-self.grid_size_half:center_indices[batch][2]+self.grid_size_half]
            heatmaps3D[batch] = self._get_heatmap_value(lookup_subset,torch.transpose(heatmaps[batch], 0,1))
        return heatmaps3D
