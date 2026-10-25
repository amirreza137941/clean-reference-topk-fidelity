"""Original reference ResNet-20 architecture and checkpoint key layout.

The shortcut-name variant is retained separately from the downsample-name
anchor to preserve both original parameter layouts and forward computations.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class OptionABasicBlock(nn.Module):
    """Block used by the previous PGD adversarial-training script."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride == 1 and in_planes == planes:
            self.shortcut = nn.Identity()
        else:
            pad = (planes - in_planes) // 2
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, pad, pad))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ProjectionBasicBlock(nn.Module):
    """Block used by the standard ResNet-20 in the attached notebook."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CIFARResNet20(nn.Module):
    def __init__(self, block_type: str, num_classes: int = 10):
        super().__init__()
        if block_type == "option_a":
            block_class = OptionABasicBlock
        elif block_type == "projection":
            block_class = ProjectionBasicBlock
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block_class, 16, blocks=3, stride=1)
        self.layer2 = self._make_layer(block_class, 32, blocks=3, stride=2)
        self.layer3 = self._make_layer(block_class, 64, blocks=3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block_class, planes: int, blocks: int, stride: int):
        layers = []
        for current_stride in [stride] + [1] * (blocks - 1):
            layers.append(block_class(self.in_planes, planes, current_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
