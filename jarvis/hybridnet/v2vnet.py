"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride = 1):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=((kernel_size-1)//2)),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.block(x))


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1,
                      padding=1),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1,
                      padding=1),
            nn.InstanceNorm3d(out_planes)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        res = self.res_branch(x)
        return self.dropout(F.relu(res + x, True))


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size,
                               stride=stride, padding=0, output_padding=0),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        return self.dropout(self.block(x))


class EncoderDecorder(nn.Module):
    def __init__(self, input_channels):
        super(EncoderDecorder, self).__init__()
        self.encoder_pool1 = Basic3DBlock(input_channels*2,
                    input_channels*4, 2,2)
        self.mid_res = Res3DBlock(input_channels*4, input_channels*4)
        self.decoder_upsample1 = Upsample3DBlock(input_channels*4,
                    input_channels*2, 2, 2)
        self.decoder_res1 = Res3DBlock(input_channels*2, input_channels*2)
        self.skip_res1 = Res3DBlock(input_channels*2, input_channels*2)

    def forward(self, x):
        res1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.mid_res(x)
        x = self.decoder_upsample1(x)
        x = self.decoder_res1(x)
        x = x + res1

        return x


class V2VNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2VNet, self).__init__()
        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, input_channels*2, 3,2),
            Res3DBlock(input_channels*2, input_channels*2)
        )
        self.encoder_decoder = EncoderDecorder(input_channels)
        self.output_layer = nn.Conv3d(input_channels*2, output_channels,
                    kernel_size=1, stride=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
