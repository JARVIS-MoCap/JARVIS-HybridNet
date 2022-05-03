"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import os

import torch
import torch.nn as nn
from torchvision import transforms

from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.utils.reprojection import ReprojectionTool


class JarvisPredictor3D(nn.Module):
    def __init__(self, cfg, weights_center_detect = 'latest',
                weights_hybridnet = 'latest', trt_mode = 'off'):
        super(JarvisPredictor3D, self).__init__()
        self.cfg = cfg

        self.centerDetect = EfficientTrack('CenterDetectInference', self.cfg,
                    weights_center_detect).model
        self.hybridNet = HybridNet('inference', self.cfg,
                    weights_hybridnet).model


        self.transform_mean = torch.tensor(self.cfg.DATASET.MEAN,
                    device = torch.device('cuda')).view(3,1,1)
        self.transform_std = torch.tensor(self.cfg.DATASET.STD,
                    device = torch.device('cuda')).view(3,1,1)
        self.bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
        self.num_cameras = self.cfg.HYBRIDNET.NUM_CAMERAS
        self.bounding_box_size = self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE

        self.reproTool = ReprojectionTool()
        self.center_detect_img_size = int(self.cfg.CENTERDETECT.IMAGE_SIZE)

        if trt_mode == 'new':
            self.compile_trt_models()

        elif trt_mode == 'previous':
            self.load_trt_models()



    def load_trt_models(self):
        # TODO: add try except here!!
        import torch_tensorrt
        transpose2D_lib_dir = os.path.join(self.cfg.PARENT_DIR, 'libs',
                    'conv_transpose2d_converter.cpython-39-x86_64-linux-gnu.so')
        transpose3D_lib_dir = os.path.join(self.cfg.PARENT_DIR, 'libs',
                    'conv_transpose3d_converter.cpython-39-x86_64-linux-gnu.so')
        torch.ops.load_library(transpose3D_lib_dir)
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(self.cfg.PARENT_DIR, 'projects',
                    self.cfg.PROJECT_NAME, 'trt-models', 'predict3D')

        # TODO: Check if files actually exist
        self.centerDetect = torch.jit.load(
                    os.path.join(trt_path, 'centerDetect.pt'))
        self.hybridNet.effTrack = torch.jit.load(
                    os.path.join(trt_path, 'keypointDetect.pt'))
        self.hybridNet.v2vNet = torch.jit.load(
                    os.path.join(trt_path, 'hybridNet.pt'))


    def compile_trt_models(self):
        # TODO: add try except here!!
        import torch_tensorrt
        transpose2D_lib_dir = os.path.join(self.cfg.PARENT_DIR, 'libs',
                    'conv_transpose2d_converter.cpython-39-x86_64-linux-gnu.so')
        transpose3D_lib_dir = os.path.join(self.cfg.PARENT_DIR, 'libs',
                    'conv_transpose3d_converter.cpython-39-x86_64-linux-gnu.so')
        torch.ops.load_library(transpose3D_lib_dir)
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(self.cfg.PARENT_DIR, 'projects',
                    self.cfg.PROJECT_NAME, 'trt-models', 'predict3D')
        os.makedirs(trt_path, exist_ok = True)

        self.centerDetect = self.centerDetect.eval().cuda()
        traced_model = torch.jit.trace(self.centerDetect,
                    [torch.randn((1, 3, 256, 256)).to("cuda")])
        self.centerDetect = torch_tensorrt.compile(traced_model,
            inputs= [torch_tensorrt.Input((self.cfg.HYBRIDNET.NUM_CAMERAS, 3,
                        self.cfg.CENTERDETECT.IMAGE_SIZE,
                        self.cfg.CENTERDETECT.IMAGE_SIZE), dtype=torch.float)],
            enabled_precisions= {torch.half},
        )
        torch.jit.save(self.centerDetect,
                    os.path.join(trt_path, 'centerDetect.pt'))


        self.hybridNet.effTrack.eval().cuda()
        traced_model = torch.jit.trace(self.hybridNet.effTrack,
                    [torch.randn((1, 3, self.bounding_box_size,
                    self.bounding_box_size)).to("cuda")])
        self.hybridNet.effTrack = torch_tensorrt.compile(traced_model,
            inputs= [torch_tensorrt.Input((self.cfg.HYBRIDNET.NUM_CAMERAS,
                        3, self.bounding_box_size, self.bounding_box_size),
                        dtype=torch.float)],
            enabled_precisions= {torch.half}
        )
        torch.jit.save(self.hybridNet.effTrack,
                    os.path.join(trt_path, 'keypointDetect.pt'))


        self.hybridNet.v2vNet.eval().cuda()
        grid_size = int(self.cfg.HYBRIDNET.ROI_CUBE_SIZE /
                    self.cfg.HYBRIDNET.GRID_SPACING)
        traced_model = torch.jit.trace(self.hybridNet.v2vNet,
                    [torch.randn((1, self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                    grid_size, grid_size, grid_size)).to("cuda")])
        self.hybridNet.v2vNet = torch_tensorrt.compile(traced_model,
        inputs= [torch_tensorrt.Input((1, self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                    grid_size, grid_size,grid_size), dtype=torch.float)],
        enabled_precisions= {torch.half}
        )
        torch.jit.save(self.hybridNet.v2vNet,
                    os.path.join(trt_path, 'hybridNet.pt'))



    def forward(self, imgs, cameraMatrices, intrinsicMatrices,
                distortionCoefficients):
        self.reproTool.cameraMatrices = cameraMatrices
        self.reproTool.intrinsicMatrices = intrinsicMatrices
        self.reproTool.distortionCoefficients = distortionCoefficients

        img_size = torch.tensor([imgs.shape[3], imgs.shape[2]],
                    device = torch.device('cuda'))

        downsampling_scale = torch.tensor([
                    imgs.shape[3] / float(self.center_detect_img_size),
                    imgs.shape[2]/float(self.center_detect_img_size)],
                    device = torch.device('cuda')).float()

        imgs_resized = transforms.functional.resize(imgs,
                    [self.center_detect_img_size,self.center_detect_img_size])
        imgs_resized = (imgs_resized - self.transform_mean) / self.transform_std
        outputs = self.centerDetect(imgs_resized)
        heatmaps_gpu = outputs[1].view(outputs[1].shape[0],
                    outputs[1].shape[1], -1)
        m = heatmaps_gpu.argmax(2).view(heatmaps_gpu.shape[0],
                    heatmaps_gpu.shape[1], 1)
        preds = torch.cat((m % outputs[1].shape[2], m // outputs[1].shape[3]),
                    dim=2)
        maxvals = heatmaps_gpu.gather(2,m)
        num_cams_detect = torch.numel(maxvals[maxvals>50])
        maxvals = maxvals/255.

        if num_cams_detect >= 2:
            center3D = self.reproTool.reconstructPoint((
                        preds.reshape(self.num_cameras,2)
                        * (downsampling_scale*2)).transpose(0,1), maxvals)
            centerHMs = self.reproTool.reprojectPoint(
                        center3D.unsqueeze(0)).int()
            centerHMs[:,0] = torch.clamp(centerHMs[:,0], self.bbox_hw,
                        img_size[0]-1-self.bbox_hw)
            centerHMs[:,1] = torch.clamp(centerHMs[:,1], self.bbox_hw,
                        img_size[1]-1-self.bbox_hw)

            imgs_cropped = torch.zeros((self.num_cameras,3,
                        self.bounding_box_size, self.bounding_box_size),
                        device = torch.device('cuda'))

            for i in range(self.num_cameras):
                imgs_cropped[i] = imgs[i,:,
                        centerHMs[i,1]-self.bbox_hw:centerHMs[i,1]+self.bbox_hw,
                        centerHMs[i,0]-self.bbox_hw:centerHMs[i,0]+self.bbox_hw]

            imgs_cropped = ((imgs_cropped - self.transform_mean)
                        / self.transform_std)

            _, _, points3D, confidences = self.hybridNet(imgs_cropped.unsqueeze(0),
                        img_size,
                        centerHMs.unsqueeze(0),
                        center3D.int().unsqueeze(0),
                        cameraMatrices.unsqueeze(0),
                        intrinsicMatrices.unsqueeze(0),
                        distortionCoefficients.unsqueeze(0))
        else:
            points3D = None
        return points3D, confidences
