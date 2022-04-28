"""
jarvis2D.py
=================
"""
import os

import torch
import torch.nn as nn
from torchvision import transforms

from jarvis.efficienttrack.efficienttrack import EfficientTrack


class JarvisPredictor2D(nn.Module):
    def __init__(self, cfg, weights_center_detect = 'latest',
                weights_keypoint_detect = 'latest', trt_mode = 'off'):
        super(JarvisPredictor2D, self).__init__()
        self.cfg = cfg

        self.centerDetect = EfficientTrack('CenterDetectInference', self.cfg,
                    weights_center_detect).model
        self.keypointDetect = EfficientTrack('KeypointDetectInference',
                    self.cfg, weights_keypoint_detect).model

        self.transform_mean = torch.tensor(self.cfg.DATASET.MEAN,
                    device = torch.device('cuda')).view(3,1,1)
        self.transform_std = torch.tensor(self.cfg.DATASET.STD,
                    device = torch.device('cuda')).view(3,1,1)
        self.bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
        self.bounding_box_size = self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE

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
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(self.cfg.PARENT_DIR, 'projects',
                    self.cfg.PROJECT_NAME, 'trt-models', 'predict2D')

        # TODO: Check if files actually exist
        self.centerDetect = torch.jit.load(
                    os.path.join(trt_path, 'centerDetect.pt'))
        self.keypointDetect = torch.jit.load(
                    os.path.join(trt_path, 'keypointDetect.pt'))



    def compile_trt_models(self):
        # TODO: add try except here!!
        import torch_tensorrt
        transpose2D_lib_dir = os.path.join(self.cfg.PARENT_DIR, 'libs',
                    'conv_transpose2d_converter.cpython-39-x86_64-linux-gnu.so')
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(self.cfg.PARENT_DIR, 'projects',
                    self.cfg.PROJECT_NAME, 'trt-models', 'predict2D')
        os.makedirs(trt_path, exist_ok = True)

        self.centerDetect = self.centerDetect.eval().cuda()
        traced_model = torch.jit.trace(self.centerDetect,
                    [torch.randn((1, 3, 256, 256)).to("cuda")])
        self.centerDetect = torch_tensorrt.compile(traced_model,
            inputs= [torch_tensorrt.Input((1, 3,
                        self.cfg.CENTERDETECT.IMAGE_SIZE,
                        self.cfg.CENTERDETECT.IMAGE_SIZE), dtype=torch.float)],
            enabled_precisions= {torch.half},
        )
        torch.jit.save(self.centerDetect,
                    os.path.join(trt_path, 'centerDetect.pt'))


        self.keypointDetect.eval().cuda()
        traced_model = torch.jit.trace(self.keypointDetect,
                    [torch.randn((1, 3, self.bounding_box_size,
                    self.bounding_box_size)).to("cuda")])
        self.keypointDetect = torch_tensorrt.compile(traced_model,
            inputs= [torch_tensorrt.Input((1, 3, self.bounding_box_size,
                        self.bounding_box_size),
                        dtype=torch.float)],
            enabled_precisions= {torch.half}
        )
        torch.jit.save(self.keypointDetect,
                    os.path.join(trt_path, 'keypointDetect.pt'))



    def forward(self, img):
        img_size = torch.tensor([img.shape[3], img.shape[2]],
                    device = torch.device('cuda'))

        downsampling_scale = torch.tensor([
                    img_size[0] / float(self.center_detect_img_size),
                    img_size[1] / float(self.center_detect_img_size)],
                    device = torch.device('cuda')).float()

        img_resized = transforms.functional.resize(img,
                    [self.center_detect_img_size,self.center_detect_img_size])
        img_resized = (img_resized - self.transform_mean) / self.transform_std
        outputs = self.centerDetect(img_resized)
        heatmaps_gpu = outputs[1].view(outputs[1].shape[0],
                    outputs[1].shape[1], -1)
        m = heatmaps_gpu.argmax(2).view(heatmaps_gpu.shape[0],
                    heatmaps_gpu.shape[1], 1)
        maxval = heatmaps_gpu.gather(2,m).squeeze()

        if maxval > 40:
            centerHM = torch.cat((m % outputs[1].shape[2], m
                        // outputs[1].shape[3]),
                        dim=2).squeeze()*downsampling_scale*2
            centerHM = centerHM.int()
            centerHM[0] = torch.clamp(centerHM[0], self.bbox_hw,
                        img_size[0]-self.bbox_hw-1)
            centerHM[1] = torch.clamp(centerHM[1], self.bbox_hw,
                        img_size[1]-self.bbox_hw-1)

            img_cropped = img[:,:,
                    centerHM[1]-self.bbox_hw:centerHM[1]+self.bbox_hw,
                    centerHM[0]-self.bbox_hw:centerHM[0]+self.bbox_hw]

            img_cropped = ((img_cropped - self.transform_mean)
                        / self.transform_std)

            outputs = self.keypointDetect(img_cropped)
            heatmaps = outputs[1].view(outputs[1].shape[0],
                        outputs[1].shape[1], -1)
            m = heatmaps.argmax(2).view(heatmaps.shape[0],
                        heatmaps.shape[1], 1)
            points2D = torch.cat((m % outputs[1].shape[2], m
                        // outputs[1].shape[3]),
                        dim=2).squeeze()*2
            maxvals = heatmaps.gather(2,m).squeeze()

            points2D = points2D+centerHM-self.bbox_hw

        else:
            points2D = None
            maxvals = None

        return points2D, maxvals
