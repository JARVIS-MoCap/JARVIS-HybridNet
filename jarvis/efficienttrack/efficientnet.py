"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import SiLU

from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_model_params,
    efficientnet_params,
    BlockArgs
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Block.

    :param block_args: Parameters for creating specific block
                       (e.g. number of filters)
    """
    def __init__(self, block_args: BlockArgs, block_idx):
        super().__init__()
        self.block_idx = block_idx
        self.padding = {1:{1:0,2:0}, 3:{1:1, 2:1}, 5:{1:2,2:2}}
        self._block_args = block_args
        self.expand_ratio = block_args.expand_ratio
        self.input_filters = self._block_args.input_filters
        self.output_filters = self._block_args.output_filters
        self.stride = self._block_args.stride
        self.num_groups = 8
        self.has_se = (self._block_args.se_ratio is not None) and (0
                       < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = (self._block_args.input_filters *
               self._block_args.expand_ratio)  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup,
                                          kernel_size=1, bias=False)
            self._gn0 = nn.InstanceNorm2d(oup)
        else:
            self._expand_conv = nn.Identity()
            self._gn0 = nn.Identity()
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        if self.block_idx < 4:
            self._depthwise_conv =  nn.Conv2d(in_channels=inp, out_channels=oup,
                                              kernel_size=k, stride = s,
                                              bias=False,
                                              padding = self.padding[k][s])

        else:
            if self._block_args.expand_ratio != 1:
                self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup,
                                              kernel_size=1, bias=False)
                self._gn0 = nn.InstanceNorm2d(oup)
            self._depthwise_conv = nn.Conv2d(
                in_channels=oup, out_channels=oup,
                groups=oup, kernel_size=k, #groups makes it depthwise
                stride=s, bias=False, padding = self.padding[k][s])
        self._gn1 = nn.InstanceNorm2d(oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1,
                int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup,
                        out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels,
                        out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup,
                    kernel_size=1, bias=False)
        self._gn2 = nn.InstanceNorm2d(final_oup)
        self._swish = SiLU()

    def forward(self, inputs, drop_connect_rate: float):
        # Expansion and Depthwise Convolution
        x = inputs
        if self.block_idx < 4:
            x = self._depthwise_conv(x)
        else:
            if self.expand_ratio != 1:
                x = self._expand_conv(inputs)
                #x = self._gn0(x)
                #x = self._swish(x)
            x = self._depthwise_conv(x)

        x = self._gn1(x)
        x = self._swish(x)


        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._gn2(x)

        # Skip connection and drop connect
        if (self.id_skip and self.stride == 1
            and self.input_filters == self.output_filters):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x



class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name
    or .from_pretrained methods

    :param block_args: Parameters for creating specific blocks
                       (e.g. number of filters)
    :param global_params: Global Parameters of the network
                          (e.g. drop_connect_rate)
    """

    def __init__(self, blocks_args: BlockArgs, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.num_groups = 8

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)
        self.drop_connect_rate = self._global_params.drop_connect_rate
        self._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                    stride=2, bias=False, padding = 1)
        self._gn0 = nn.InstanceNorm2d(out_channels)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for idx,block_args in enumerate(self._blocks_args):

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters = round_filters(block_args.input_filters,
                            self._global_params),
                output_filters = round_filters(block_args.output_filters,
                            self._global_params),
                num_repeat = round_repeats(block_args.num_repeat,
                            self._global_params)
            )

            self._blocks.append(MBConvBlock(block_args, idx))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                            input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, idx))

            self._swish = SiLU()


    def forward(self, inputs):
        bs = inputs.size(0)
        x = self._swish(self._gn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name,
                                                      override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000,
                        in_channels=3):
        model = cls.from_name(model_name,
                    override_params = {'num_classes': num_classes})
        if in_channels != 3:
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = nn.Conv2d(in_channels, out_channels,
                        kernel_size=3, stride=2, bias=False, padding = 1)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: '
                        + ', '.join(valid_models))
