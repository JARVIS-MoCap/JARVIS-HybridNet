"""
model.py
========
EfficientTrack torch module.
"""

import torch
from torch import nn
from torchvision.ops.boxes import nms as nms_torch

from lib.hybridnet.efficienttrack.efficientnet import EfficientNet as EffNet
from .utils import MaxPool2dStaticSamePadding
from lib.utils.utils import Swish, MemoryEfficientSwish


class EfficientTrackBackbone(nn.Module):
    """
    EfficientTrack torch module. Uses group normalization instead of batch norm.

    :param num_classes: Number of object classes
    :type num_classes: int
    :param compound_coef: Compound Coefficient of the network.
                          Smaller coefficients correspond to smaller
                          (both in memory and FLOPs) networks.
    :type compound_coef: int
    :param onnx_export: Select if model will be exported to onnx format. If True
        regular swish implementation will be used
    :type onnx_export: bool, optional
    """
    def __init__(self, cfg, compound_coef=0, output_channels = 1,
                 onnx_export = False, **kwargs):
        super(EfficientTrackBackbone, self).__init__()
        self.num_groups = 8
        self.cfg = cfg
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [88, 112, 160, 224, 288, 384, 384, 384, 384]
        self.fpn_cell_repeats = [4, 4, 6, 7, 7, 8, 8, 8, 8]
        self.final_layer_sizes = [88, 112, 160, 192, 224, 288, 288, 384, 384]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [24, 40, 112],
            1: [24, 40, 112],
            2: [24, 48, 120],
            3: [32, 48, 136],
            4: [32, 56, 160],
            5: [40, 64, 176],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7, onnx_export = onnx_export)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.backbone_net = EfficientNet(
                    self.backbone_compound_coef[compound_coef])
        self.backbone_net.model.set_swish(not onnx_export)

        self.swish = MemoryEfficientSwish()
        self.upsample3 = nn.Upsample(scale_factor = 4, mode = 'nearest')
        self.upsample2 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.weights_cat = nn.Parameter(torch.ones(3), requires_grad=True)
        self.weights_relu = nn.ReLU()
        self.first_conv = SeparableConvBlock(
                    self.fpn_num_filters[self.compound_coef],
                    self.final_layer_sizes[self.compound_coef],True)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=self.final_layer_sizes[self.compound_coef],
            out_channels=self.final_layer_sizes[self.compound_coef],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.gn1 = nn.GroupNorm(self.num_groups,
                                self.final_layer_sizes[self.compound_coef])
        self.final_conv1 = nn.Conv2d(
            in_channels = self.final_layer_sizes[self.compound_coef],
            out_channels = output_channels,
            kernel_size = 3,
            padding=1,
            bias = False)
        self.final_conv2 = nn.Conv2d(
            in_channels = self.final_layer_sizes[self.compound_coef],
            out_channels = output_channels,
            kernel_size = 3,
            padding=1,
            bias = False)


    def forward(self, inputs):
        p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)
        x3 = self.upsample3(features[2])
        x2 = self.upsample2(features[1])

        weight = self.weights_relu(self.weights_cat)
        weight = weight / (torch.sum(weight, dim=0) + 0.0001)
        x1 = weight[0]*features[0]+weight[1]*x2+weight[2]*x3
        res1 = self.first_conv(x1)
        res2 = self.deconv1(res1)
        res2 = self.gn1(res2)
        res2 = self.swish(res2)

        res1 = self.final_conv1(res1)
        res2 = self.final_conv2(res2)


        return [res1, res2]



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
    :param onnx_export: Select if model will be exported to onnx format. If True
        regular swish implementation will be used
    :type onnx_export: bool, optional
    """
    def __init__(self, in_channels, out_channels = None, norm = True,
                 activation = False, onnx_export = False):
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
            self.gn = nn.GroupNorm(self.num_groups, out_channels)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

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
    :param onnx_export: Select if model will be exported to onnx format. If True
        regular swish implementation will be used
    :type onnx_export: bool, optional
    """
    def __init__(self, in_channels, out_channels = None, norm = True,
                 activation = False, onnx_export = False):
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
            self.gn = nn.GroupNorm(self.num_groups, out_channels)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

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
    :param onnx_export: Select if model will be exported to onnx format. If True
        regular swish implementation will be used
    :type onnx_export: bool, optional
    :param attention: TODO
    :type attention: bool, optional
    :param use_p8: Specifies wether an extra FPN stage is used
    :type use_p8: bool, optional
    """
    def __init__(self, num_channels, conv_channels, first_time=False,
                 epsilon=1e-4, onnx_export=False, attention=True, use_p8=False):
        super(BiFPN, self).__init__()
        self.register_buffer('epsilon', torch.tensor(epsilon))
        self.register_buffer('num_groups', torch.tensor(8))
        self.register_buffer('first_time', torch.tensor(first_time))

        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels,
                                           onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels,
                                           onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels,
                                           onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels,
                                           onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels,
                                             onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels,
                                             onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels,
                                             onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels,
                                             onnx_export=onnx_export)
        if use_p8:
            self.conv7_up = SeparableConvBlock(num_channels,
                                               onnx_export=onnx_export)
            self.conv8_down = SeparableConvBlock(num_channels,
                                                 onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[2], num_channels, 1),
                nn.GroupNorm(self.num_groups, num_channels),
            )
            self.p4_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[1], num_channels, 1),
                nn.GroupNorm(self.num_groups, num_channels),
            )
            self.p3_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[0], num_channels, 1),
                nn.GroupNorm(self.num_groups, num_channels),
            )

            self.p5_to_p6 = nn.Sequential(
                nn.Conv2d(conv_channels[2], num_channels, 1),
                nn.GroupNorm(self.num_groups, num_channels),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )
            if use_p8:
                self.p7_to_p8 = nn.Sequential(
                    MaxPool2dStaticSamePadding(3, 2)
                )
            self.p4_down_channel_2 = nn.Sequential(
                nn.Conv2d(conv_channels[1], num_channels, 1),
                nn.GroupNorm(self.num_groups, num_channels),
            )
            self.p5_down_channel_2 = nn.Sequential(
                nn.Conv2d(conv_channels[2], num_channels, 1),
                nn.GroupNorm(self.num_groups, num_channels),
            )

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

        self.attention = attention

    def forward(self, inputs):
        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        #print ('Eps', self.epsilon.get_device())
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1]
                    * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1]
                    * self.p5_upsample(p6_up)))


        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1]
                    * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1]
                    * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2]
                        * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2]
                        * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2]
                        * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1]
                    * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        if self.use_p8:
            # P8_0 to P8_2

            # Connections for P7_0 and P8_0 to P7_1 respectively
            p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
        else:
            # P7_0 to P7_2

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.use_p8:
            # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
            p7_out = self.conv7_down(
                self.swish(p7_in + p7_up + self.p7_downsample(p6_out)))

            # Connections for P8_0 and P7_2 to P8_2
            p8_out = self.conv8_down(self.swish(p8_in
                        + self.p8_downsample(p7_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        else:
            # Connections for P7_0 and P6_2 to P7_2
            p7_out = self.conv7_down(self.swish(p7_in
                        + self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out



class EfficientNet(nn.Module):
    """
    EfficientNet modified to return the intermediate feature maps needed for the
    BiFPN module.

    :param compound_coef: Compound Coefficient (see EfficientNet base class for
        details)
    :type compound_coef: int
    :return: Feature Maps at 1/16, 1/32, 1/64 of the original resolution.
    """
    def __init__(self, compound_coef):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}')
        del model._conv_head
        del model._gn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

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
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if self.save_idxs[idx+1]:
                feature_maps.append(x)
        return feature_maps
