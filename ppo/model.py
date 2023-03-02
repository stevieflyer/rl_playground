import torch
from torch import nn
import numpy as np


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        # convert x to torch.Tensor if it's still a numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        out = self.fc(x)
        return out



