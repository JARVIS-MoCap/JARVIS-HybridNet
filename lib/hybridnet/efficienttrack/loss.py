"""
loss.py
============
MSE Heatmap Loss for EfficientTrack.
"""

import torch
import torch.nn as nn

class HeatmapLoss(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()

    def forward(self, outputs, heatmaps):
        heatmaps_losses = []
        for idx in range(len(outputs)):
            assert outputs[idx].size() == heatmaps[idx].size()
            loss = ((outputs[idx] - heatmaps[idx])**2)
            loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
            heatmaps_losses.append(loss)
        return heatmaps_losses
