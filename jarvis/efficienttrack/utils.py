"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import itertools
import numpy as np
import cv2
import collections
import re
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_


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


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    """
    Calculate and round number of filters based on depth multiplier.

    :param filters: Number of filters in baseline model
    :type filters: int
    :param global_params: Global Parameters containing scaling factors
    :return: scaled number of filters
    :rtype: int
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor/2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """
    Calculate and round number of block repeats based on depth multiplier.

    :param filters: Number of block repeats in baseline model
    :type filters: int
    :param global_params: Global Parameters containing scaling factors
    :return: scaled number of block repeats
    :rtype: int
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p: float, training: bool):
    """
    Drop Connect implementation. Drops random activations, with probability p
    and renormalizes activations.

    :param inputs: Input acitvations
    :type inputs: torch.tensor
    :param p: Drop probability
    :type p: float
    :param training: Specifiec wether network is in taining or inference mode,
        only drops activation during training
    :type training: bool
    :return: Dropout activations
    :rtype: torch.tensor
    """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype,
                device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def efficientnet_params(model_name):
    """
    Map EfficientNet model name to parameter coefficients.

    :param model_name: Name of the model, formated like "efficientnet-b<x>"
                       with <x> being the compound coefficient.
    :return: Dictionary with scaling factors and dropout probability
    :rtype: dict
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (0.5, 0.5, 256, 0.2),
        'efficientnet-b1': (1.0, 1.0, 256, 0.2),
        'efficientnet-b2': (1.0, 1.1, 240, 0.2),
        'efficientnet-b3': (1.1, 1.2, 260, 0.3),
        'efficientnet-b4': (1.2, 1.4, 300, 0.3),
        'efficientnet-b5': (1.4, 1.8, 380, 0.4),
        'efficientnet-b6': (1.6, 2.2, 456, 0.4),
        'efficientnet-b7': (1.8, 2.6, 528, 0.5),
        'efficientnet-b8': (2.0, 3.1, 600, 0.5),
        'efficientnet-b9': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """
    Block Decoder for readability, straight from the official TensorFlow
    repository
    """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s'][0]))

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of
                            block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.2, image_size=None,
                 num_classes=1000):
    """
    Creates a efficientnet model.

    :param width_coefficient: Multiplier for number of conv channels
    :type width_coefficient: flaot, optional
    :param depth_coefficient: Multiplier for number of MBConvBlocks
    :type depth_coefficient: flaot, optional
    :param dropout_rate: Dropout probability for final layer
    :type dropout_rate: float, optional
    :param drop_connect_rate: Dropout probability for MBConvBlocks
    :type drop_connect_rate: float, optional
    :param image_size: image size
    :type image_size: int, optional
    :param num_classes: number of classes for final fc layer
    :type num_classes: int, optional
    :return: Block Args and Global Parameters
    """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
                    width_coefficient=w, depth_coefficient=d, dropout_rate=p,
                    image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s'
                    % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not
        # included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params
