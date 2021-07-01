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
    def __init__(self, cfg, intrinsic_paths, extrinsic_paths, lookup_path = None):
        super(ReprojectionLayer, self).__init__()
        self.cfg = cfg
        self.root_dir = cfg.DATASET.DATASET_DIR
        self.reproTool = ReprojectionTool('T', self.root_dir, intrinsic_paths, extrinsic_paths)
        if lookup_path == None:
            self.register_buffer('reproLookup', torch.from_numpy(self.__create_lookup()))
        else:
            self.reproLookup = torch.from_numpy(np.load(lookup_path).astype('int16')).permute(3,0,1,2,4)
            #self.register_buffer('reproLookup', torch.from_numpy(np.load(lookup_path).astype('int16')).permute(3,0,1,2,4))

        self.register_buffer('offset', torch.tensor([self.cfg.VORTEX.GRID_DIM_X[0], self.cfg.VORTEX.GRID_DIM_Y[0], self.cfg.VORTEX.GRID_DIM_Z[0]]))
        self.register_buffer('grid_spacing', torch.tensor(self.cfg.VORTEX.GRID_SPACING))
        self.register_buffer('boxsize', torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE))
        self.register_buffer('grid_size', torch.tensor(self.cfg.VORTEX.ROI_CUBE_SIZE/self.cfg.VORTEX.GRID_SPACING).int())
        self.register_buffer('num_cameras',  torch.tensor(self.reproTool.num_cameras))
        self.register_buffer('grid_size',  torch.tensor(self.boxsize/self.grid_spacing).int())
        self.register_buffer('grid_size_half',  torch.tensor(self.boxsize/self.grid_spacing/2).int())

        self.ii,self.xx,self.yy,self.zz = torch.meshgrid(torch.arange(self.num_cameras).cuda(),
                                                         torch.arange(self.grid_size).cuda(),
                                                         torch.arange(self.grid_size).cuda(),
                                                         torch.arange(self.grid_size).cuda())
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def __create_lookup(self):
        x = np.arange(self.cfg.VORTEX.GRID_DIM_X[0], self.cfg.VORTEX.GRID_DIM_X[1], self.cfg.VORTEX.GRID_SPACING)
        y = np.arange(self.cfg.VORTEX.GRID_DIM_Y[0], self.cfg.VORTEX.GRID_DIM_Y[1], self.cfg.VORTEX.GRID_SPACING)
        z = np.arange(self.cfg.VORTEX.GRID_DIM_Z[0], self.cfg.VORTEX.GRID_DIM_Z[1], self.cfg.VORTEX.GRID_SPACING)
        reproLookup = np.zeros((len(x),len(y),len(z), self.reproTool.num_cameras, 2))

        def par_test(i):
            lookup = np.zeros((len(y),len(z), self.reproTool.num_cameras, 2))
            print (i)
            for j in range(len(y)):
                for k in range(len(z)):
                    lookup[j,k] = self.reproTool.reprojectPoint([x[i],y[j],z[k]])
            return i, lookup

        result = Parallel(n_jobs=24)(delayed(par_test)(i) for i in range(len(x)))
        for element in result:
            reproLookup[element[0]] = element[1]/2
        np.save('lookup.npy', reproLookup)
        reproLookup = torch.from_numpy(reproLookup)
        return reproLookup


    def __get_heatmap_value(self, lookup, heatmaps):
        lookup = lookup.cuda().long()
        heatmaps = heatmaps.flatten(2)
        self.starter.record()
        #print (heatmaps.flatten(2).shape)
        #print (lookup.shape)
        outs = torch.mean(heatmaps[:,self.ii,lookup[self.ii,self.xx,self.yy,self.zz]], dim = 1)
        #print (outs.shape)
        self.ender.record()

        return outs

    def forward(self, heatmaps, center):
        center_indices = ((center-self.offset)/self.grid_spacing).int()
        heatmaps3D = torch.cuda.FloatTensor(heatmaps.shape[0], 23, self.grid_size, self.grid_size, self.grid_size)
        for batch in range(heatmaps.shape[0]):
            lookup_subset = self.reproLookup[:,center_indices[batch][0]-self.grid_size_half:center_indices[batch][0]+self.grid_size_half,
                                             center_indices[batch][1]-self.grid_size_half:center_indices[batch][1]+self.grid_size_half,
                                             center_indices[batch][2]-self.grid_size_half:center_indices[batch][2]+self.grid_size_half]
            lookup_subset2 = torch.cuda.FloatTensor(12, 104,104,104)
            print (lookup_subset.shape)
            for i in range(12):
                print (i)
                lookup_subset2[i] = lookup_subset[i,:,:,:,1]*512+lookup_subset[i,:,:,:,0]
            heatmaps3D[batch] = self.__get_heatmap_value(lookup_subset2,torch.transpose(heatmaps[batch], 0,1))
        return heatmaps3D
