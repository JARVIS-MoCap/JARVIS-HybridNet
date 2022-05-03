"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import torch
from torch import nn

from .efficientnet import EfficientNet as EffNet
from torch.nn import SiLU

import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple, List


class EfficientTrackBackbone(nn.Module):
    """
    EfficientTrack torch module. Uses group normalization instead of batch norm.

    :param num_classes: Number of object classes
    :type num_classes: int
    :param model_size: Scaling factor of the network (small, medium, large)..
    :type model_size: str
    """
    def __init__(self, cfg, model_size='small', output_channels = 1, **kwargs):
        super(EfficientTrackBackbone, self).__init__()
        self.num_groups = 8
        self.cfg = cfg
        self.model_size = model_size

        if model_size == 'small':
            self.backbone_compound_coef = 0
            self.fpn_num_filters = 56
            self.fpn_cell_repeats = 3
            self.final_layer_sizes = 64
            self.conv_channel_coef = [16, 24, 56]
        elif model_size == 'medium':
            self.backbone_compound_coef = 1
            self.fpn_num_filters = 88
            self.fpn_cell_repeats = 4
            self.final_layer_sizes = 88
            self.conv_channel_coef = [24, 40, 112]
        elif model_size == 'large':
            self.backbone_compound_coef = 3
            self.fpn_num_filters = 160
            self.fpn_cell_repeats = 6
            self.final_layer_sizes = 160
            self.conv_channel_coef = [24, 48, 120]


        #self.fpn_num_filters = [56,88, 112, 160, 224, 288, 384, 384, 384, 384]
        #self.fpn_cell_repeats = [3,4, 4, 6, 7, 7, 8, 8, 8, 8]
        #self.final_layer_sizes = [64,88, 112, 160, 192, 224, 288, 288, 384, 384]
        #self.pyramid_levels = [5,5, 5, 5, 5, 5, 5, 5, 5, 6]

        # conv_channel_coef = {
        #     # the channels of P3/P4/P5.
        #     0: [16, 24, 56],
        #     1: [24, 40, 112],
        #     2: [24, 40, 112],
        #     3: [24, 48, 120],
        #     4: [32, 48, 136],
        #     5: [32, 56, 160],
        #     6: [40, 64, 176],
        #     7: [72, 200, 576],
        #     8: [72, 200, 576],
        #     9: [80, 224, 640],
        # }

        self.bifpn = nn.ModuleList([BiFPN_first(self.fpn_num_filters,
                    self.conv_channel_coef)] + \
                    [BiFPN(self.fpn_num_filters,
                    self.conv_channel_coef)
                    for _ in range(1,self.fpn_cell_repeats)])

        self.backbone_net = EfficientNet(self.backbone_compound_coef)

        self.swish = SiLU()
        self.upsample3 = nn.Upsample(scale_factor = 4, mode = 'nearest')
        self.upsample2 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.weights_cat = nn.Parameter(torch.ones(3), requires_grad=True)
        self.weights_relu = nn.Softplus()
        self.first_conv = SeparableConvBlock(
                    self.fpn_num_filters,
                    self.final_layer_sizes,True)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=self.final_layer_sizes,
            out_channels=output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)

        self.gn1 = nn.InstanceNorm2d(self.final_layer_sizes)
        self.final_conv1 = nn.Conv2d(
            in_channels = self.final_layer_sizes,
            out_channels = output_channels,
            kernel_size = 3,
            padding=1,
            bias = False)
        self.final_conv2 = nn.Conv2d(
            in_channels = self.final_layer_sizes,
            out_channels = output_channels,
            kernel_size = 1,
            padding=0,
            bias = False)



    def forward(self, inputs: torch.Tensor):
        features = self.backbone_net(inputs)
        for bifpn in self.bifpn:
            features = bifpn(features)

        x3 = self.upsample3(features[2])
        x2 = self.upsample2(features[1])


        weight = self.weights_relu(self.weights_cat)
        weight = weight / (torch.sum(weight, dim=0) + 0.0001)
        x1 = weight[0]*features[0]+weight[1]*x2+weight[2]*x3
        res1 = self.first_conv(x1)
        res2 = self.deconv1(res1)
        res1 = self.final_conv1(res1)

        return (res1, res2)



class RegularConvBlock(nn.Module):
    """
    Regular ConvBlock with optional normalization and activation.

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels, equal to input channels if
        not specified
    :type out_channels: int, optional
    :param norm: Select if group normalization is applied
    :type norm: bool, optional
    :param activation: Select if swish is applied after convolution
    :type activation: bool, optional
    """
    def __init__(self, in_channels, out_channels = None, norm = True,
                 activation = False):
        super(RegularConvBlock, self).__init__()
        self.num_groups = 8
        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size=3,
            stride=1,
            padding =1)

        self.norm = norm
        if self.norm:
            self.gn1 = nn.InstanceNorm2d(out_channels)

        self.activation = activation
        if self.activation:
            self.swish = SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.gn(x)
        if self.activation:
            x = self.swish(x)
        return x



class SeparableConvBlock(nn.Module):
    """
    Depthwise seperable ConvBlock with optional normalization and activation.

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels, equal to input channels if
        not specified
    :type out_channels: int, optional
    :param norm: Select if group normalization is applied
    :type norm: bool, optional
    :param activation: Select if swish is applied after convolution
    :type activation: bool, optional
    """
    def __init__(self, in_channels, out_channels = None, norm = True,
                 activation = False):
        super(SeparableConvBlock, self).__init__()
        self.num_groups = 8
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = 1,
            groups = in_channels,
            bias = False,
            padding = 1)
        self.pointwise_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0)

        self.norm = norm
        if self.norm:
            self.gn = nn.InstanceNorm2d(out_channels)

        self.activation = activation
        self.swish = SiLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.gn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    BiFPN (Weighted Birdirectional Feature Pyramid Network) implementation.

    :param num_channels: Number of convolutional channels for all convolutions
        used in BiFPN
    :type num_channels: int
    :param conv_channels: Number of xonvolutional channels of the different
                          input stages
    :type conv_channels: int
    :param first_time: Specifies wether this is the first BiFPN block after the
        EffNet backbone, if yes some tranisition convs are added
    :type first_time: bool, optional
    :param epsilon: small constant for numerical stability of the weight
                    normalization
    :type epsilon: float, optional
    :param attention: TODO
    :type attention: bool, optional
    """
    def __init__(self, num_channels, conv_channels,
                 epsilon=1e-4, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.num_groups = 8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels)
        self.conv5_up = SeparableConvBlock(num_channels)
        self.conv4_up = SeparableConvBlock(num_channels)
        self.conv3_up = SeparableConvBlock(num_channels)
        self.conv4_down = SeparableConvBlock(num_channels)
        self.conv5_down = SeparableConvBlock(num_channels)
        self.conv6_down = SeparableConvBlock(num_channels)
        self.conv7_down = SeparableConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(2, 2)
        self.p5_downsample = nn.MaxPool2d(2, 2)
        self.p6_downsample = nn.MaxPool2d(2, 2)
        self.p7_downsample = nn.MaxPool2d(2, 2)

        self.swish = SiLU()

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p3_in = inputs[0]
        p4_in = inputs[1]
        p5_in = inputs[2]
        p6_in = inputs[3]
        p7_in = inputs[4]

        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1]
                    * self.p6_upsample(p7_in)))

        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1]
                    * self.p5_upsample(p6_up)))

        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1]
                    * self.p4_upsample(p5_up)))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1]
                    * self.p3_upsample(p4_up)))

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2]
                        * self.p4_downsample(p3_out)))

        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2]
                        * self.p5_downsample(p4_out)))

        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2]
                        * self.p6_downsample(p5_out)))

        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1]
                    * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class BiFPN_first(nn.Module):
    """
    BiFPN_first (Weighted Birdirectional Feature Pyramid Network) implementation.

    :param num_channels: Number of convolutional channels for all convolutions
        used in BiFPN
    :type num_channels: int
    :param conv_channels: Number of xonvolutional channels of the different
                          input stages
    :type conv_channels: int
    :param first_time: Specifies wether this is the first BiFPN block after the
        EffNet backbone, if yes some tranisition convs are added
    :type first_time: bool, optional
    :param epsilon: small constant for numerical stability of the weight
                    normalization
    :type epsilon: float, optional
    :param attention: TODO
    :type attention: bool, optional
    """
    def __init__(self, num_channels, conv_channels,
                 epsilon=1e-4, attention=True):
        super(BiFPN_first, self).__init__()
        self.epsilon = epsilon
        self.num_groups = 8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels)
        self.conv5_up = SeparableConvBlock(num_channels)
        self.conv4_up = SeparableConvBlock(num_channels)
        self.conv3_up = SeparableConvBlock(num_channels)
        self.conv4_down = SeparableConvBlock(num_channels)
        self.conv5_down = SeparableConvBlock(num_channels)
        self.conv6_down = SeparableConvBlock(num_channels)
        self.conv7_down = SeparableConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(2, 2)
        self.p5_downsample = nn.MaxPool2d(2, 2)
        self.p6_downsample = nn.MaxPool2d(2, 2)
        self.p7_downsample = nn.MaxPool2d(2, 2)

        self.swish = SiLU()

        self.p5_down_channel = nn.Sequential(
            nn.Conv2d(conv_channels[2], num_channels, 1),
            nn.InstanceNorm2d(num_channels))
        self.p4_down_channel = nn.Sequential(
            nn.Conv2d(conv_channels[1], num_channels, 1),
            nn.InstanceNorm2d(num_channels))
        self.p3_down_channel = nn.Sequential(
            nn.Conv2d(conv_channels[0], num_channels, 1),
            nn.InstanceNorm2d(num_channels))

        self.p5_to_p6 = nn.Sequential(
            nn.Conv2d(conv_channels[2], num_channels, 1),
            nn.InstanceNorm2d(num_channels),
            nn.MaxPool2d(2, 2))
        self.p6_to_p7 = nn.Sequential(
            nn.MaxPool2d(2, 2))
        self.p4_down_channel_2 = nn.Sequential(
            nn.Conv2d(conv_channels[1], num_channels, 1),
            nn.InstanceNorm2d(num_channels))
        self.p5_down_channel_2 = nn.Sequential(
            nn.Conv2d(conv_channels[2], num_channels, 1),
            nn.InstanceNorm2d(num_channels))

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()


    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = inputs[0]
        p4 = inputs[1]
        p5 = inputs[2]

        p6_in = self.p5_to_p6(p5)
        p7_in = self.p6_to_p7(p6_in)
        p3_in = self.p3_down_channel(p3)
        p4_in = self.p4_down_channel(p4)
        p5_in = self.p5_down_channel(p5)

        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1]
                    * self.p6_upsample(p7_in)))

        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1]
                    * self.p5_upsample(p6_up)))

        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1]
                    * self.p4_upsample(p5_up)))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1]
                    * self.p3_upsample(p4_up)))

        p4_in = self.p4_down_channel_2(p4)
        p5_in = self.p5_down_channel_2(p5)

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2]
                        * self.p4_downsample(p3_out)))

        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2]
                        * self.p5_downsample(p4_out)))

        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2]
                        * self.p6_downsample(p5_out)))

        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1]
                    * self.p7_downsample(p6_out)))

        return (p3_out, p4_out, p5_out, p6_out, p7_out)



class EfficientNet(nn.Module):
    """
    EfficientNet modified to return the intermediate feature maps needed for the
    BiFPN module.

    :param compound_coef: Scaling parameter for the model
    :type model_size: int
    :return: Feature Maps at 1/16, 1/32, 1/64 of the original resolution.
    """
    def __init__(self, compound_coef):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}')
        self.model = model
        self.drop_connect_rate = self.model._global_params.drop_connect_rate
        self.save_idxs = []
        ignore_first = True
        last_idx = 0
        for idx, block in enumerate(self.model._blocks):
            if ignore_first and block._depthwise_conv.stride == (2,2):
                ignore_first = False
                self.save_idxs.append(False)
            else:
                self.save_idxs.append(block._depthwise_conv.stride == (2,2))
                if block._depthwise_conv.stride == (2,2):
                    last_idx = idx-1
        self.model._blocks = self.model._blocks[:last_idx+1]

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._gn0(x)
        x = self.model._swish(x)
        feature_maps = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if self.save_idxs[idx+1]:
                feature_maps.append(x)
        return feature_maps
