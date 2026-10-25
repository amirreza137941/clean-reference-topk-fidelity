"""Original CIFAR projection backbone and dual checkpoint container.

Parameter names and forward computations are retained for existing weights.
The Top-k wrapper uses ONLY the normalized robust_branch of a dual checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class ProjectionBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CIFARResNet20(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, 1)
        self.layer2 = self._make_layer(32, 3, 2)
        self.layer3 = self._make_layer(64, 3, 2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for current_stride in [stride] + [1] * (blocks - 1):
            layers.append(ProjectionBasicBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class DualBackboneDefense(nn.Module):
    def __init__(
        self,
        selector_temperature: float = 1.0,
        primary_margin: float = 1.0,
    ):
        super().__init__()
        self.robust_branch = CIFARResNet20(num_classes=10)
        self.set2_branch = CIFARResNet20(num_classes=10)
        self.selector_temperature = float(selector_temperature)
        self.primary_margin = float(primary_margin)
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def compose_logits(
        self,
        robust_logits: torch.Tensor,
        set2_logits: torch.Tensor,
    ) -> torch.Tensor:
        primary = robust_logits.argmax(dim=1)
        hard = F.one_hot(primary, num_classes=robust_logits.shape[1]).to(
            robust_logits.dtype
        )
        soft = F.softmax(robust_logits / self.selector_temperature, dim=1)
        straight_through = hard + soft - soft.detach()
        primary_base = (set2_logits * hard).sum(dim=1)
        maximum_base = set2_logits.max(dim=1).values
        boost = (
            (maximum_base - primary_base + self.primary_margin)
            .clamp_min(self.primary_margin)
            .detach()
        )
        return set2_logits + straight_through * boost[:, None]

    def forward_all(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.normalize(x)
        robust_logits = self.robust_branch(normalized)
        set2_logits = self.set2_branch(normalized)
        final_logits = self.compose_logits(robust_logits, set2_logits)
        return robust_logits, set2_logits, final_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)[2]
