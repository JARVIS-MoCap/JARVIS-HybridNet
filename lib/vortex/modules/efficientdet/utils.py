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




class BBoxTransform(nn.Module):
    """
    TODO: Description for what this does

    :param anchors: [batchsize, boxes, (y1, x1, y2, x2)]
    :type anchors: list, optional
    :param regression: [batchsize, boxes, (dy, dx, dh, dw)]
    :type anchors: list
    :return: TODO
    """
    def forward(self, anchors, regression):
        """
        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):
    """
    TODO: Clip bounding boxes to  imagesize

    :param boxes: TODO
    :type boxes: TODO
    :param img: TODO
    :type anchors: TODO
    :return: Clipped Bounding Boxes
    """
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


class Anchors(nn.Module):
    """
    Generates multiscale anchor boxes.

    :param anchor_scale: Number representing the scale of size of the base
                         anchor to the feature stride 2^level.
    :type anchor_scale: float
    :param anchor_configs: A dictionary with keys as the levels of anchors
                           and values as a list of anchor configuration.
    :type anchor_configs: dict
    :param pyramid_levels: List of pyramid levels to create anchors for
    :type pyramid_levels: list, optional
    :return: A numpy array with shape [N, 4], which stacks anchors on all
             feature levels.
    """
    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes

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
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)



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
    Initialize weigts of EfficientDet using kaiman uniform initialization and
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
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in ori_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_imgs = [(img[..., ::-1].astype(np.float32) / 255 - mean) / std for img in framed_imgs]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    """
    Creates labeled BoundingBox Predictions from newtork output.

    :param x: batch that was used as input for the network
    :type x: torch.tensor
    :param anchors: Bounding box anchors
    :type anchors: torch.tensor
    :param regression: output of regression head
    :type regression: torch.tensor
    :param classification: output of classification head
    :type classification: torch.tensor
    :param regressBoxes: TODO
    :type regressBoxes: TODO
    :param clipBoxes: TODO
    :type clipBoxes: TODO
    :param threshold: classifcation likelihood threshold
    :type threshold: float
    :param iou_threshold: Intersection-over-Union threshold for bbox predictions
    :type iou_threshold: float
    :return: Dict containing final roi predictions and corresponding class_ids
        and their likelihood scores
    :rtype: dict
    """
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
    return out
