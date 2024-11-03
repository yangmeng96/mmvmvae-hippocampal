import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class ClfOdorRats(nn.Module):
    def __init__(self, original_dim):  # , z_dim, size, nfilter=64, nfilter_max=1024, **kwargs):
        super().__init__()
        self.original_dim = original_dim
        self.blocks = nn.Sequential(
            nn.Linear(self.original_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        out = self.blocks(x)
        return out