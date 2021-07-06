"""
utils.py
========
Utility functions for EfficientDet.
"""

import itertools
import numpy as np
from typing import Union
import math
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
from torchvision.ops.boxes import batched_nms


class MaxPool2dStaticSamePadding(nn.Module):
    """
    Tensorflow like static paddding for BiFPN fuse connections
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.left = 0
        self.right = 1
        self.bottom = 1
        self.top = 0

    def forward(self, x):
        x = F.pad(x, [self.left, self.right, self.top, self.bottom])

        x = self.pool(x)
        return x



class CustomDataParallel(nn.DataParallel):
    """
    Force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.

    :param num_gpus: Number of GPUs the data is split to
    :type num_gpus: int
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus
        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')
        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                [inputs[1][i][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True) for i in range(2)])
                for device_idx in range(len(devices))], [kwargs] * len(devices)



def invert_affine(metas: Union[float, list, tuple], preds):
    """
    Invert affine transform applied to input data to get correct predictions
    """
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds



def variance_scaling_(tensor, gain=1.):
    """
    Initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/VarianceScaling
    :param tensor: TODO
    :type tensor: torch.tensor
    :param gain: gain for scaling
    :type gain: float
    :return: TODO
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)



def init_weights(model):
    """
    Initialize weights of EfficientTrack using kaiman uniform initialization and
    variance scaling.
    """
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()



def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h



def preprocess(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Preprocess image from path. see :func:`~preprocess_img` for details.
    """
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    return preprocess_img(ori_imgs, max_size, mean, std)



def preprocess_img(ori_imgs, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Preprocess image by resizing, padding and z-scoring.

    :param ori_imgs: Images to be preprocessed
    :type ori_imgs: list
    :param max_size: size of image to be returned
    :type max_size: int, optional
    :param mean: mean for z-scoring
    :type mean: (float,float,float), optional
    :param std: standard deviation for z-scoring
    :type std: (float,float,float), optional
    :return: Original images, preprocessed images and metas that can be used by
        :func:`~invert_affine`.
    """
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas
