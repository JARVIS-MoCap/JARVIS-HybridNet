import torch
import torch.nn as nn

class ReprojectionLayer(nn.Module):
    def __init__(self, cfg, num_cameras = None):
        super(ReprojectionLayer, self).__init__()
        self.cfg = cfg

        self.grid_spacing = self.cfg.HYBRIDNET.GRID_SPACING
        self.boxsize = self.cfg.HYBRIDNET.ROI_CUBE_SIZE
        self.grid_size = int(self.cfg.HYBRIDNET.ROI_CUBE_SIZE /
                    self.cfg.HYBRIDNET.GRID_SPACING)

        if num_cameras:
            self.num_cameras = num_cameras
        else:
            self.num_cameras = self.cfg.HYBRIDNET.NUM_CAMERAS

        self.grid = torch.zeros((int(self.grid_size/2), int(self.grid_size/2),
                                 int(self.grid_size/2),3))
        half_gridsize = int(self.grid_size/2/2)
        for i in range(int(self.grid_size/2)):
            for j in range(int(self.grid_size/2)):
                for k in range(int(self.grid_size/2)):
                    self.grid[i,j,k] = torch.tensor([i - half_gridsize,
                                                     j - half_gridsize,
                                                     k - half_gridsize])
        self.grid = self.grid.cuda()
        self.grid = self.grid * self.grid_spacing*2
        self.heatmap_size = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2+2)


    def reprojectPoints(self, x, cameraMatrices, intrinsicMatrices,
                distortionCoefficients, centerHM):
      intrinsicMatrices = intrinsicMatrices.permute(1,2,0)
      distortionCoefficients = distortionCoefficients.permute(1,2,0)
      centerHM = centerHM.permute(1,0)

      ones = torch.ones([x.shape[0], x.shape[1], x.shape[2],1],
                device=torch.device('cuda'))
      x = torch.cat((x,ones),3)

      partial_all = torch.matmul(x.view(1,-1,4), cameraMatrices).view(-1,
                int(self.grid_size/2), int(self.grid_size/2),
                int(self.grid_size/2),3).permute(1,2,3,4,0)

      val1 = (partial_all[:,:,:,0] / partial_all[:,:,:,2]
                - intrinsicMatrices[2,0])
      val2 = (partial_all[:,:,:,1] / partial_all[:,:,:,2]
                - intrinsicMatrices[2,1])
      r2 = (torch.square(val1 / intrinsicMatrices[0,0])
                + torch.square(val2 / intrinsicMatrices[1,1]))
      distort = 1 + (distortionCoefficients[0,0] +
                distortionCoefficients[0,1] * r2) * r2
      val1 = val1 * distort + intrinsicMatrices[2,0]
      val2 = val2 * distort + intrinsicMatrices[2,1]

      val1 = torch.clamp(val1, centerHM[0]-(self.heatmap_size-1),
                centerHM[0]+self.heatmap_size-2)-centerHM[0]+self.heatmap_size-1
      val2 = torch.clamp(val2, centerHM[1]-(self.heatmap_size-1),
                centerHM[1]+self.heatmap_size-2)-centerHM[1]+self.heatmap_size-1

      val1 = nn.functional.interpolate(val1.permute(3,0,1,2).view(1,-1,int(
                self.grid_size/2),int(self.grid_size/2),int(self.grid_size/2)),
                size=(self.grid_size,self.grid_size,self.grid_size),
                mode='trilinear').view(self.num_cameras,self.grid_size,
                self.grid_size, self.grid_size).permute(1,2,3,0)

      val2 = nn.functional.interpolate(val2.permute(3,0,1,2).view(1,-1,int(
                self.grid_size/2),int(self.grid_size/2),int(self.grid_size/2)),
                size=(self.grid_size,self.grid_size,self.grid_size),
                mode='trilinear').view(self.num_cameras,self.grid_size,
                self.grid_size, self.grid_size).permute(1,2,3,0)

      res = ((val2 / 2).int() * self.heatmap_size
            + (val1 / 2).int()).permute(3,0,1,2).long()

      return res


    def _get_heatmap_value(self, heatmaps, grid, cameraMatrices,
                intrinsicMatrices, distortionCoefficients, centerHM):
        reproPoints = self.reprojectPoints(grid,cameraMatrices,
                    intrinsicMatrices, distortionCoefficients, centerHM)

        num_joints =  heatmaps.shape[0];
        num_cameras =  heatmaps.shape[1];
        heatmap_size =  heatmaps.shape[2];
        grid_size =  reproPoints.shape[2];
        cam_offset = torch.arange(0,heatmap_size*heatmap_size*num_cameras,
                    heatmap_size*heatmap_size, device = torch.device('cuda'))

        heatmaps = heatmaps.flatten(1);
        reproPoints = (reproPoints.flatten(1).transpose(1,0)
                    + cam_offset).transpose(1,0).flatten()
        outs = torch.mean(torch.index_select(heatmaps, 1, reproPoints).view(
                    (num_joints,num_cameras,grid_size,grid_size,grid_size)),
                    dim = 1)

        return outs


    def forward(self, heatmaps, center, centerHM, cameraMatrices,
                intrinsicMatrices, distortionCoefficients):
        # for batch in range(heatmaps.shape[0]):
        grid = self.grid+center[0]
        heatmaps3D = self._get_heatmap_value(torch.transpose(
                    heatmaps[0], 0,1), grid, cameraMatrices[0],
                    intrinsicMatrices[0], distortionCoefficients[0],
                    centerHM[0]).unsqueeze(0)

        return heatmaps3D
