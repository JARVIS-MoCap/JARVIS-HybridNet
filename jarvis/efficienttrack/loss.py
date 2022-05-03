"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import torch.nn as nn

class HeatmapLoss(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()

    def forward(self, outputs, heatmaps):
        heatmaps_losses = []
        for idx in range(len(outputs)):
            loss = ((outputs[idx] - heatmaps[idx])**2)
            loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
            heatmaps_losses.append(loss)
        return heatmaps_losses
