import torch
import torch.nn as nn
from pykeops.torch import LazyTensor

class KernelDistance(nn.Module):
    def __init__(self, sigma=1.0):
        super(KernelDistance, self).__init__()
        self.sigma = sigma

    def forward(self, points):
        # points: (B, N, 3)
        X_i = LazyTensor(points[:, :, None, :])  # (B, N, 1, 3)
        X_j = LazyTensor(points[:, None, :, :])  # (B, 1, N, 3)
        D_ij = ((X_i - X_j) ** 2).sum(-1)         # (B, N, N)
        K = (-D_ij / (2 * self.sigma ** 2)).exp()   # (B, N, N)
        return K