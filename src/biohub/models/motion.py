import torch
from torch import nn

from biohub.features.motion import MOTION_FEATURES

__all__ = ['MOTION_FEATURES', 'MotionResidual']


class MotionResidual(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
