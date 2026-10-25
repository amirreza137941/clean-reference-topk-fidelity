"""Pixel-space model adapters; normalization happens inside each network."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class NormalizedModel(nn.Module):
    """Retain the original model/mean/std checkpoint keys."""

    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class RobustPixelModel(nn.Module):
    """Expose only the original dual checkpoint's normalized robust branch."""

    def __init__(self, dual: nn.Module):
        super().__init__()
        self.dual = dual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dual.robust_branch(self.dual.normalize(x))


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze parameters and batch statistics, while allowing input gradients."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
