"""
utils.py
===============
Utility Funcions used across the whole HybridNet library
"""

import torch
from torch import nn

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


#MultiGPU Stuff for HybridNet, currently disabled because untested and probaby broken
# class CustomDataParallel(nn.DataParallel):
#     """
#     force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
#     """
#
#     def __init__(self, module, num_gpus, mode):
#         super().__init__(module)
#         self.num_gpus = num_gpus
#         self.mode = mode
#
#     def scatter(self, inputs, kwargs, device_ids):
#         # More like scatter and data prep at the same time. The point is we prep the data in such a way
#         # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
#         devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
#         splits = inputs[0].shape[0] // self.num_gpus
#
#         if splits == 0:
#             raise Exception('Batchsize must be greater than num_gpus.')
#         return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
#                  inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
#                  inputs[2][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
#                  inputs[3][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
#                  inputs[4][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
#                 for device_idx in range(len(devices))], \
#                [kwargs] * len(devices)

#MultiGPU Stuff for EfficientTrack, currently disabled because untested and probaby broken
# class CustomDataParallel(nn.DataParallel):
#     """
#     Force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
#
#     :param num_gpus: Number of GPUs the data is split to
#     :type num_gpus: int
#     """
#
#     def __init__(self, module, num_gpus):
#         super().__init__(module)
#         self.num_gpus = num_gpus
#
#     def scatter(self, inputs, kwargs, device_ids):
#         # More like scatter and data prep at the same time. The point is we prep the data in such a way
#         # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
#         devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
#         splits = inputs[0].shape[0] // self.num_gpus
#         if splits == 0:
#             raise Exception('Batchsize must be greater than num_gpus.')
#         return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
#                 [inputs[1][i][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True) for i in range(2)])
#                 for device_idx in range(len(devices))], [kwargs] * len(devices)
