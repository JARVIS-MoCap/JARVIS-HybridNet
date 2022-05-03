"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = 0
        for i,gt_batch in enumerate(gt):
            for j, gt_single in enumerate(gt_batch):
                if torch.sum(gt_single) > 1:
                    loss += torch.mean(((pred[i][j] - gt_single)**2))
        return loss
