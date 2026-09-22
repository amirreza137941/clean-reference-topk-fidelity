#!/usr/bin/env python3
"""Evaluate logit-aligned and confidence-gated Top-1 + Set-2 defenses.

Proposed inference pipeline for every input x:

1. The standard ResNet-20 selects reference classes c1 and c2.
2. The adversarially trained ResNet-20 supplies gradients for the exact two
   boundaries of interest: c1 versus every other class (Top-1), and the weaker
   member of {c1, c2} versus the strongest outside class (Set-2).
3. One or more bounded projected signed input steps are applied.
4. Isotropic Gaussian input noise is added.
5. The adversarially trained ResNet-20 produces the final logits.
6. In ``proposed80`` only, the two reference logits receive the smallest
   validation-calibrated bias that reaches the requested clean Set-2 target;
   c1 may receive a second bias to protect Top-1.
7. In ``proposed_gated``, that bias is enabled per sample only when the
   standard Top-2 is stable across noisy views and both classes remain within
   a calibrated Top-k support region of the robust model. Gate parameters are
   selected using clean and PGD-proxy validation data, never test data.

Only two outcomes are optimized and reported:
  * top1: true-label Top-1 accuracy.
  * set2: exact unordered match with the clean standard model's Top-2 set.

The targeted-step parameters and Gaussian sigma are calibrated on a held-out
training validation subset, not on the reported test subset. Every black-box
attack is run with two distinct goals: breaking Top-1 and breaking the clean
reference Set-2. This avoids both test-set tuning and the misleading practice
of claiming Set-2 robustness from a Top-1-only attack.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
STANDARD_CIFAR10_STD = (0.2471, 0.2435, 0.2616)  # attached notebook
ROBUST_CIFAR10_STD = (0.2470, 0.2435, 0.2616)    # previous PGD-AT script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model Top-2 + Gaussian defense evaluation on CIFAR-10"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets/cifar-10-batches-py"),
        help="cifar-10-batches-py itself, or its parent directory",
    )
    parser.add_argument(
        "--standard-weights",
        type=Path,
        default=Path(
            "./models/resnet20/cifar10/"
            "resnet20_cifar10.pth"
        ),
    )
    parser.add_argument(
        "--robust-weights",
        type=Path,
        default=Path(
            "./models/resnet20/mixed_pgd_at/"
            "resnet20_mixed_pgd_best_composite_weights.pt"
        ),
        help="Balanced clean/PGD checkpoint from mixed adversarial fine-tuning",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./models/resnet20/"
            "top1_set2_gated_evaluation"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--n-examples", type=int, default=1000)

    # Proposed targeted strengthening step, matching the attached Top2 design.
    parser.add_argument("--strength-alpha", type=float, default=0.004)
    parser.add_argument("--strength-epsilon", type=float, default=0.03)
    parser.add_argument(
        "--strength-steps",
        type=int,
        default=2,
        help="Projected ascent steps used by the targeted strengthening stage",
    )
    parser.add_argument(
        "--top1-weight",
        type=float,
        default=1.5,
        help="Weight of the exact c1-versus-all Top-1 margin",
    )
    parser.add_argument(
        "--set2-weight",
        type=float,
        default=2.5,
        help="Weight of the weakest-in-set versus strongest-outside margin",
    )

    # By default, tune only inference-time defense parameters on a validation
    # subset reconstructed from the mixed-AT training split. The selection rule
    # maximizes clean Set-2 subject to a small allowed Top-1 drop.
    parser.add_argument("--calibration-examples", type=int, default=500)
    parser.add_argument("--training-val-size", type=int, default=5000)
    parser.add_argument("--training-split-seed", type=int, default=42)
    parser.add_argument(
        "--no-tune-strength",
        action="store_true",
        help="Use only --strength-alpha/steps and the two explicit weights",
    )
    parser.add_argument(
        "--strength-alpha-grid", type=float, nargs="+", default=[0.003, 0.004, 0.005]
    )
    parser.add_argument(
        "--strength-step-grid", type=int, nargs="+", default=[1, 2]
    )
    parser.add_argument(
        "--top1-weight-grid", type=float, nargs="+", default=[1.0, 1.5, 2.0]
    )
    parser.add_argument(
        "--set2-weight-grid", type=float, nargs="+", default=[1.5, 2.5, 4.0]
    )
    parser.add_argument(
        "--tune-max-top1-drop",
        type=float,
        default=0.5,
        help="Max clean Top-1 drop (percentage points) allowed during Set-2 tuning",
    )

    # Gaussian sigma is calibrated to reduce the proposed sigma=0 clean set2
    # by these percentage-point budgets.
    parser.add_argument(
        "--sigma-drops",
        type=float,
        nargs="+",
        default=[0.5, 2.0],
        help="Validation Set-2 drop budgets; 0.5 favors clean, 2 favors noise",
    )
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--sigma-search-iters", type=int, default=15)
    parser.add_argument("--sigma-trials", type=int, default=3)
    parser.add_argument("--sigma-tolerance", type=float, default=0.3)

    # Minimal post-logit alignment used only by the proposed80 variant. A
    # default target of 85% gives a generalization buffer above the user's
    # requested 80% test Set-2 goal.
    parser.add_argument("--target-clean-set2", type=float, default=85.0)
    parser.add_argument("--alignment-sigma-drop", type=float, default=0.5)
    parser.add_argument("--logit-bias-max", type=float, default=20.0)
    parser.add_argument("--logit-bias-grid-size", type=int, default=81)
    parser.add_argument("--logit-bias-calibration-mc", type=int, default=3)
    parser.add_argument(
        "--logit-bias-max-top1-drop",
        type=float,
        default=0.5,
        help="Allowed validation Top-1 drop versus the unaligned proposed model",
    )

    # Adaptive gate. The gate validates the standard model's Top-2 selection
    # using noisy-view stability and the robust model's support rank before any
    # logit bias is applied.
    parser.add_argument("--gated-target-clean-set2", type=float, default=82.0)
    parser.add_argument("--gate-views", type=int, default=3)
    parser.add_argument("--gate-view-sigma", type=float, default=0.015)
    parser.add_argument(
        "--gate-stability-grid", type=float, nargs="+", default=[0.0, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--gate-support-topk-grid",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10],
    )
    parser.add_argument(
        "--gate-bias-grid",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    )
    parser.add_argument("--gate-calibration-mc", type=int, default=3)
    parser.add_argument("--gate-max-clean-top1-drop", type=float, default=0.5)
    parser.add_argument("--proxy-pgd-steps", type=int, default=10)
    parser.add_argument("--proxy-pgd-alpha", type=float, default=2.0 / 255.0)

    parser.add_argument("--attack-epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument(
        "--attacks", nargs="+", default=["Square", "NES", "SignHunt"],
        choices=["Square", "NES", "SignHunt"],
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument(
        "--attack-goals",
        nargs="+",
        choices=["top1", "set2"],
        default=["top1", "set2"],
        help="Run each attack separately against Top-1 and/or Set-2",
    )
    parser.add_argument(
        "--attack-eot",
        type=int,
        default=1,
        help="Average this many stochastic defense calls per attacker query",
    )
    parser.add_argument(
        "--eval-mc",
        type=int,
        default=1,
        help="Average this many stochastic calls for final reported logits",
    )
    parser.add_argument(
        "--defenses",
        nargs="+",
        default=[
            "robust", "gaussian0p5", "proposed0p5", "proposed80",
            "proposed_gated",
        ],
        help=(
            "Subset of standard, robust, targeted, gaussian<drop>, "
            "proposed<drop>, proposed80, proposed_gated; decimal points use p "
            "(for example proposed0p5), or all"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: 100 images, Square+SignHunt, budgets 100/500",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Calibrate parameters and report clean metrics without running attacks",
    )
    parser.add_argument(
        "--resume-results",
        action="store_true",
        help="Skip attack combinations already present in attack_results.csv",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_torchvision_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "cifar-10-batches-py" and (path / "data_batch_1").is_file():
        return path.parent
    if (path / "cifar-10-batches-py" / "data_batch_1").is_file():
        return path
    raise FileNotFoundError(
        "CIFAR-10 Python batches not found. Expected data_batch_1 under "
        f"'{path}' or '{path / 'cifar-10-batches-py'}'."
    )


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
                lambda x: F.pad(
                    x[:, :, ::stride, ::stride], (0, 0, 0, 0, pad, pad)
                )
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


class NormalizedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mean: Sequence[float] = CIFAR10_MEAN,
        std: Sequence[float] = STANDARD_CIFAR10_STD,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(loaded: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(loaded, Mapping):
        raise TypeError("Checkpoint is not a mapping/state_dict")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = loaded.get(key)
        if isinstance(candidate, Mapping) and candidate:
            loaded = candidate
            break
    if not loaded or not all(isinstance(k, str) for k in loaded):
        raise ValueError("Could not find a valid model state_dict in checkpoint")
    state = dict(loaded)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def load_resnet20(
    checkpoint_path: Path,
    device: torch.device,
    fallback_std: Sequence[float],
) -> Tuple[nn.Module, str]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {checkpoint_path}")
    state = _extract_state_dict(_torch_load(checkpoint_path, device))

    # Some common CIFAR ResNet-20 implementations call the projection branch
    # ``downsample`` (torchvision convention), while the attached notebook calls
    # the identical branch ``shortcut``.  The tensors and forward computation
    # are the same; normalize only the state-dict names before architecture
    # detection/loading.
    state = {
        key.replace(".downsample.", ".shortcut."): value
        for key, value in state.items()
    }

    wrapped = "model.conv1.weight" in state
    if wrapped:
        body_state = {
            key[len("model."):]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
    else:
        body_state = state

    projection_key = "layer2.0.shortcut.0.weight"
    block_type = "projection" if projection_key in body_state else "option_a"
    body = CIFARResNet20(block_type=block_type, num_classes=10)
    model = NormalizedModel(body, CIFAR10_MEAN, fallback_std)

    if wrapped:
        missing, unexpected = model.load_state_dict(state, strict=False)
        meaningful_missing = [key for key in missing if key not in {"mean", "std"}]
    else:
        missing, unexpected = body.load_state_dict(body_state, strict=False)
        meaningful_missing = list(missing)
    if meaningful_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {checkpoint_path}. "
            f"Missing={meaningful_missing}; unexpected={list(unexpected)}"
        )

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, block_type


def load_balanced_subset(
    data_dir: Path,
    n_examples: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples <= 0 or n_examples > 10000 or n_examples % 10 != 0:
        raise ValueError("--n-examples must be divisible by 10 and between 10 and 10000")
    root = resolve_torchvision_root(data_dir)
    dataset = torchvision.datasets.CIFAR10(
        root=str(root), train=False, download=False, transform=transforms.ToTensor()
    )
    # Seeded stratified random sampling avoids the positional bias caused by
    # taking the first N examples of each class from the CIFAR-10 test file.
    rng = np.random.default_rng(seed)
    targets = np.asarray(dataset.targets)
    per_class = n_examples // 10
    indices: List[int] = []
    for label in range(10):
        class_indices = np.flatnonzero(targets == label)
        rng.shuffle(class_indices)
        indices.extend(class_indices[:per_class].tolist())
    rng.shuffle(indices)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=n_examples,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)


def stratified_validation_indices(
    targets: Sequence[int], val_size: int, seed: int
) -> List[int]:
    """Reconstruct the held-out split used by mixed-PGD fine-tuning."""
    if val_size <= 0 or val_size >= len(targets) or val_size % 10 != 0:
        raise ValueError(
            "--training-val-size must be divisible by 10 and smaller than 50000"
        )
    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    per_class = val_size // 10
    validation_indices: List[int] = []
    for class_id in range(10):
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)
        validation_indices.extend(class_indices[:per_class].tolist())
    rng.shuffle(validation_indices)
    return validation_indices


def load_calibration_subset(
    data_dir: Path,
    n_examples: int,
    training_val_size: int,
    split_seed: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a balanced subset of the training validation split."""
    if n_examples <= 0 or n_examples % 10 != 0:
        raise ValueError("--calibration-examples must be positive and divisible by 10")
    if n_examples > training_val_size:
        raise ValueError("--calibration-examples cannot exceed --training-val-size")
    root = resolve_torchvision_root(data_dir)
    dataset = torchvision.datasets.CIFAR10(
        root=str(root), train=True, download=False, transform=transforms.ToTensor()
    )
    validation_indices = stratified_validation_indices(
        dataset.targets, training_val_size, split_seed
    )
    per_class = n_examples // 10
    selected: List[int] = []
    counts = [0] * 10
    for index in validation_indices:
        label = int(dataset.targets[index])
        if counts[label] < per_class:
            selected.append(index)
            counts[label] += 1
        if all(count == per_class for count in counts):
            break
    if any(count != per_class for count in counts):
        raise RuntimeError(f"Could not build calibration subset; counts={counts}")
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=n_examples,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)


class NoDefense:
    def __init__(self, model: nn.Module):
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)


class GaussianInputNoise:
    def __init__(self, model: nn.Module, sigma: float):
        self.model = model
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noisy = torch.clamp(x + self.sigma * torch.randn_like(x), 0.0, 1.0)
        with torch.no_grad():
            return self.model(noisy)


def apply_reference_logit_bias(
    logits: torch.Tensor,
    reference_top2: torch.Tensor,
    set2_bias: float,
    top1_bias: float,
    sample_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Boost both reference logits, with an optional extra boost for c1."""
    if set2_bias == 0.0 and top1_bias == 0.0:
        return logits
    adjusted = logits.clone()
    if sample_gate is None:
        gate = torch.ones(
            (len(logits), 1), device=adjusted.device, dtype=adjusted.dtype
        )
    else:
        gate = sample_gate.to(device=adjusted.device, dtype=adjusted.dtype).view(-1, 1)
    if set2_bias != 0.0:
        common_source = gate.expand(-1, 2) * float(set2_bias)
        adjusted.scatter_add_(1, reference_top2, common_source)
    if top1_bias != 0.0:
        c1 = reference_top2[:, :1]
        c1_source = gate * float(top1_bias)
        adjusted.scatter_add_(1, c1, c1_source)
    return adjusted


def maximum_reference_rank(
    logits: torch.Tensor, reference_top2: torch.Tensor
) -> torch.Tensor:
    """Return the worse (larger) 1-based rank of c1 and c2 per sample."""
    order = logits.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(
        1, logits.shape[1] + 1, device=logits.device, dtype=order.dtype
    ).view(1, -1).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks.gather(1, reference_top2).max(dim=1).values


class CrossModelTop2Defense:
    """Strengthen the standard model's c1 and unordered {c1,c2} boundaries."""

    def __init__(
        self,
        standard_model: nn.Module,
        robust_model: nn.Module,
        alpha: float = 0.01,
        epsilon: float = 0.03,
        steps: int = 1,
        sigma: float = 0.0,
        top1_weight: float = 2.0,
        set2_weight: float = 1.0,
        set2_logit_bias: float = 0.0,
        top1_logit_bias: float = 0.0,
        gate_enabled: bool = False,
        gate_views: int = 3,
        gate_view_sigma: float = 0.015,
        gate_stability_threshold: float = 0.75,
        gate_support_topk: int = 4,
    ):
        self.standard_model = standard_model
        self.robust_model = robust_model
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.sigma = float(sigma)
        self.top1_weight = float(top1_weight)
        self.set2_weight = float(set2_weight)
        self.set2_logit_bias = float(set2_logit_bias)
        self.top1_logit_bias = float(top1_logit_bias)
        self.gate_enabled = bool(gate_enabled)
        self.gate_views = int(gate_views)
        self.gate_view_sigma = float(gate_view_sigma)
        self.gate_stability_threshold = float(gate_stability_threshold)
        self.gate_support_topk = int(gate_support_topk)
        if self.top1_weight < 0.0:
            raise ValueError("top1_weight must be non-negative")
        if self.set2_weight < 0.0:
            raise ValueError("set2_weight must be non-negative")
        if self.steps < 1:
            raise ValueError("strengthening steps must be >= 1")
        if self.set2_logit_bias < 0.0 or self.top1_logit_bias < 0.0:
            raise ValueError("logit biases must be non-negative")
        if self.gate_views < 0 or self.gate_support_topk < 2:
            raise ValueError("gate_views must be >= 0 and support_topk >= 2")

    def _objective(
        self, logits: torch.Tensor, reference_top2: torch.Tensor
    ) -> torch.Tensor:
        selected_logits = logits.gather(1, reference_top2)
        z1 = selected_logits[:, 0]

        # Exact Top-1 boundary: c1 must beat c2 as well as all outside classes.
        not_c1 = torch.ones_like(logits, dtype=torch.bool)
        not_c1.scatter_(1, reference_top2[:, :1], False)
        strongest_not_c1 = logits.masked_fill(
            ~not_c1, float("-inf")
        ).max(dim=1).values

        # Exact unordered Set-2 boundary. The min operation adaptively sends the
        # Set-2 gradient to whichever reference member is currently at risk.
        outside = torch.ones_like(logits, dtype=torch.bool)
        outside.scatter_(1, reference_top2, False)
        weakest_selected = selected_logits.min(dim=1).values
        strongest_outside = logits.masked_fill(
            ~outside, float("-inf")
        ).max(dim=1).values

        top1_margin = z1 - strongest_not_c1
        set2_margin = weakest_selected - strongest_outside
        return self.top1_weight * top1_margin + self.set2_weight * set2_margin

    def _reference_and_stability(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            reference_top2 = self.standard_model(x).topk(2, dim=1).indices
            if not self.gate_enabled or self.gate_views == 0:
                stability = torch.ones(len(x), device=x.device)
            else:
                reference_set = reference_top2.sort(dim=1).values
                agreement = torch.ones(len(x), device=x.device)
                for _ in range(self.gate_views):
                    noisy_view = torch.clamp(
                        x + self.gate_view_sigma * torch.randn_like(x), 0.0, 1.0
                    )
                    view_set = self.standard_model(noisy_view).topk(
                        2, dim=1
                    ).indices.sort(dim=1).values
                    agreement += view_set.eq(reference_set).all(dim=1).float()
                stability = agreement / float(self.gate_views + 1)
        return reference_top2, stability

    def forward_components(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pre-bias logits, references, stability, and raw support rank."""
        reference_top2, stability = self._reference_and_stability(x)

        base = x.detach()
        with torch.no_grad():
            # Measure support before the targeted step can promote the selected
            # classes; otherwise the gate would create its own evidence.
            raw_support_rank = maximum_reference_rank(
                self.robust_model(base), reference_top2
            )
        strengthened = base.clone()
        for _ in range(self.steps):
            with torch.enable_grad():
                x_for_grad = strengthened.detach().requires_grad_(True)
                objective = self._objective(
                    self.robust_model(x_for_grad), reference_top2
                )
                gradient = torch.autograd.grad(
                    objective.sum(),
                    x_for_grad,
                    only_inputs=True,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            strengthened = strengthened + self.alpha * gradient.sign()
            delta = (strengthened - base).clamp(-self.epsilon, self.epsilon)
            strengthened = torch.clamp(base + delta, 0.0, 1.0).detach()

        if self.sigma > 0.0:
            strengthened = torch.clamp(
                strengthened + self.sigma * torch.randn_like(strengthened),
                0.0,
                1.0,
            )
        with torch.no_grad():
            final_logits = self.robust_model(strengthened)
        return final_logits, reference_top2, stability, raw_support_rank

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        final_logits, reference_top2, stability, support_rank = (
            self.forward_components(x)
        )
        if self.gate_enabled:
            gate = (
                (stability >= self.gate_stability_threshold)
                & (support_rank <= self.gate_support_topk)
            )
        else:
            gate = torch.ones(len(x), device=x.device, dtype=torch.bool)
        with torch.no_grad():
            return apply_reference_logit_bias(
                final_logits,
                reference_top2,
                set2_bias=self.set2_logit_bias,
                top1_bias=self.top1_logit_bias,
                sample_gate=gate,
            )


def averaged_query(defense: Callable, x: torch.Tensor, trials: int) -> torch.Tensor:
    if trials < 1:
        raise ValueError("Monte Carlo/EOT trials must be >= 1")
    total = None
    for _ in range(trials):
        logits = defense(x).detach()
        total = logits if total is None else total + logits
    return total / float(trials)


def compute_logits_batched(
    defense: Callable,
    images: torch.Tensor,
    batch_size: int,
    mc_trials: int = 1,
) -> torch.Tensor:
    parts = []
    for low in range(0, len(images), batch_size):
        batch = images[low : low + batch_size]
        parts.append(averaged_query(defense, batch, mc_trials).cpu())
    return torch.cat(parts, dim=0)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=1).indices
    return 100.0 * topk.eq(labels[:, None]).any(dim=1).float().mean().item()


def set_match_topk(logits: torch.Tensor, reference: torch.Tensor, k: int) -> float:
    current_topk = logits.topk(k, dim=1).indices.sort(dim=1).values
    reference_topk = reference.topk(k, dim=1).indices.sort(dim=1).values
    return 100.0 * current_topk.eq(reference_topk).all(dim=1).float().mean().item()


def all_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
) -> Dict[str, float]:
    labels = labels.cpu()
    reference_logits = reference_logits.cpu()
    logits = logits.cpu()
    return {
        "top1": topk_accuracy(logits, labels, 1),
        "set2": set_match_topk(logits, reference_logits, 2),
    }


def mean_clean_set2(
    defense: Callable,
    images: torch.Tensor,
    reference_logits: torch.Tensor,
    batch_size: int,
    trials: int,
) -> float:
    values = []
    for _ in range(trials):
        logits = compute_logits_batched(defense, images, batch_size, mc_trials=1)
        values.append(set_match_topk(logits, reference_logits, 2))
    return float(np.mean(values))


def tune_strength_parameters(
    standard_model: nn.Module,
    robust_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    batch_size: int,
    epsilon: float,
    alpha_values: Sequence[float],
    step_values: Sequence[int],
    top1_weight_values: Sequence[float],
    set2_weight_values: Sequence[float],
    max_top1_drop: float,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, float]]:
    """Maximize validation Set-2 while protecting robust-model clean Top-1."""
    robust_metrics = all_metrics(
        compute_logits_batched(
            NoDefense(robust_model), images, batch_size, mc_trials=1
        ),
        labels,
        reference_logits,
    )
    top1_floor = robust_metrics["top1"] - max_top1_drop
    rows: List[Dict[str, object]] = []
    for alpha, steps, top1_weight, set2_weight in itertools.product(
        alpha_values, step_values, top1_weight_values, set2_weight_values
    ):
        defense = CrossModelTop2Defense(
            standard_model=standard_model,
            robust_model=robust_model,
            alpha=float(alpha),
            epsilon=epsilon,
            steps=int(steps),
            sigma=0.0,
            top1_weight=float(top1_weight),
            set2_weight=float(set2_weight),
        )
        metrics = all_metrics(
            compute_logits_batched(defense, images, batch_size, mc_trials=1),
            labels,
            reference_logits,
        )
        feasible = metrics["top1"] >= top1_floor
        rows.append(
            {
                "alpha": float(alpha),
                "steps": int(steps),
                "epsilon": epsilon,
                "top1_weight": float(top1_weight),
                "set2_weight": float(set2_weight),
                "validation_top1": metrics["top1"],
                "validation_set2": metrics["set2"],
                "top1_floor": top1_floor,
                "feasible": int(feasible),
                "selected": 0,
            }
        )

    feasible_rows = [row for row in rows if row["feasible"] == 1]
    pool = feasible_rows if feasible_rows else rows
    # Primary criterion is Set-2. Ties prefer Top-1, then less perturbation.
    selected = max(
        pool,
        key=lambda row: (
            float(row["validation_set2"]),
            float(row["validation_top1"]),
            -float(row["alpha"]) * int(row["steps"]),
        ),
    )
    selected["selected"] = 1
    parameters = {
        "alpha": float(selected["alpha"]),
        "steps": float(selected["steps"]),
        "top1_weight": float(selected["top1_weight"]),
        "set2_weight": float(selected["set2_weight"]),
    }
    return parameters, rows, robust_metrics


def tune_reference_logit_bias(
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    target_set2: float,
    max_top1_drop: float,
    bias_max: float,
    grid_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, float]]:
    """Find the smallest validation-only logit bias meeting Set-2 and Top-1."""
    base_logits = base_logits.cpu()
    labels = labels.cpu()
    reference_logits = reference_logits.cpu()
    reference_top2 = reference_logits.topk(2, dim=1).indices
    baseline = all_metrics(base_logits, labels, reference_logits)
    top1_floor = baseline["top1"] - max_top1_drop
    bias_values = np.linspace(0.0, bias_max, grid_size).tolist()
    rows: List[Dict[str, object]] = []

    for set2_bias, top1_bias in itertools.product(bias_values, bias_values):
        adjusted = apply_reference_logit_bias(
            base_logits,
            reference_top2,
            set2_bias=float(set2_bias),
            top1_bias=float(top1_bias),
        )
        metrics = all_metrics(adjusted, labels, reference_logits)
        feasible = (
            metrics["set2"] >= target_set2
            and metrics["top1"] >= top1_floor
        )
        rows.append(
            {
                "set2_logit_bias": float(set2_bias),
                "top1_logit_bias": float(top1_bias),
                "validation_top1": metrics["top1"],
                "validation_set2": metrics["set2"],
                "target_set2": target_set2,
                "top1_floor": top1_floor,
                "feasible": int(feasible),
                "selected": 0,
            }
        )

    feasible_rows = [row for row in rows if row["feasible"] == 1]
    if not feasible_rows:
        best = max(
            rows,
            key=lambda row: (
                float(row["validation_set2"]),
                float(row["validation_top1"]),
            ),
        )
        raise RuntimeError(
            "No logit-bias pair met the requested clean Set-2/Top-1 constraints. "
            f"Best grid point reached top1={best['validation_top1']:.2f}, "
            f"set2={best['validation_set2']:.2f}. Increase --logit-bias-max "
            "or relax --logit-bias-max-top1-drop."
        )

    # Minimal total bias is the primary selection rule. Metrics only break ties.
    selected = min(
        feasible_rows,
        key=lambda row: (
            float(row["set2_logit_bias"]) + float(row["top1_logit_bias"]),
            -float(row["validation_top1"]),
            -float(row["validation_set2"]),
            float(row["top1_logit_bias"]),
        ),
    )
    selected["selected"] = 1
    parameters = {
        "set2_logit_bias": float(selected["set2_logit_bias"]),
        "top1_logit_bias": float(selected["top1_logit_bias"]),
        "validation_top1": float(selected["validation_top1"]),
        "validation_set2": float(selected["validation_set2"]),
    }
    return parameters, rows, baseline


def pgd_proxy_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_reference_top2: torch.Tensor,
    goal: str,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """Generate validation-only PGD proxies for gate selection."""
    clean = images.detach()
    adversarial = torch.clamp(
        clean + torch.empty_like(clean).uniform_(-epsilon, epsilon), 0.0, 1.0
    )
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model(adversarial)
            if goal == "top1":
                attack_objective = -top1_attack_margin(logits, labels)
            elif goal == "set2":
                attack_objective = -set2_attack_margin(
                    logits, clean_reference_top2
                )
            else:
                raise ValueError(f"Unknown proxy goal: {goal}")
            gradient = torch.autograd.grad(
                attack_objective.sum(), adversarial, only_inputs=True
            )[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = torch.clamp(clean + delta, 0.0, 1.0)
    return adversarial.detach()


def compute_gate_features_batched(
    defense: CrossModelTop2Defense,
    images: torch.Tensor,
    batch_size: int,
    mc_trials: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average pre-bias logits/stability and derive robust support ranks."""
    logit_parts, reference_parts, stability_parts, rank_parts = [], [], [], []
    for low in range(0, len(images), batch_size):
        batch = images[low : low + batch_size]
        total_logits = None
        total_stability = None
        reference_top2 = None
        for _ in range(mc_trials):
            logits, current_reference, stability, current_support_rank = (
                defense.forward_components(batch)
            )
            total_logits = logits if total_logits is None else total_logits + logits
            total_stability = (
                stability
                if total_stability is None
                else total_stability + stability
            )
            if reference_top2 is None:
                reference_top2 = current_reference
                support_rank = current_support_rank
        mean_logits = total_logits / float(mc_trials)
        mean_stability = total_stability / float(mc_trials)
        logit_parts.append(mean_logits.detach().cpu())
        reference_parts.append(reference_top2.detach().cpu())
        stability_parts.append(mean_stability.detach().cpu())
        rank_parts.append(support_rank.detach().cpu())
    return (
        torch.cat(logit_parts),
        torch.cat(reference_parts),
        torch.cat(stability_parts),
        torch.cat(rank_parts),
    )


def tune_gated_alignment(
    clean_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    top1_proxy_features: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    set2_proxy_features: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    labels: torch.Tensor,
    clean_reference_logits: torch.Tensor,
    target_clean_set2: float,
    max_clean_top1_drop: float,
    stability_values: Sequence[float],
    support_topk_values: Sequence[int],
    bias_values: Sequence[float],
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, float]]:
    """Tune the gate on clean and proxy-adversarial validation examples."""
    labels = labels.cpu()
    clean_reference_logits = clean_reference_logits.cpu()
    clean_logits, clean_refs, clean_stability, clean_ranks = clean_features
    top1_logits, top1_refs, top1_stability, top1_ranks = top1_proxy_features
    set2_logits, set2_refs, set2_stability, set2_ranks = set2_proxy_features
    clean_baseline = all_metrics(clean_logits, labels, clean_reference_logits)
    top1_floor = clean_baseline["top1"] - max_clean_top1_drop
    rows: List[Dict[str, object]] = []

    for stability_threshold, support_topk, bias in itertools.product(
        stability_values, support_topk_values, bias_values
    ):
        def evaluate_features(logits, refs, stability, ranks):
            gate = (
                (stability >= float(stability_threshold))
                & (ranks <= int(support_topk))
            )
            adjusted = apply_reference_logit_bias(
                logits,
                refs,
                set2_bias=float(bias),
                top1_bias=0.0,
                sample_gate=gate,
            )
            metrics = all_metrics(adjusted, labels, clean_reference_logits)
            return metrics, 100.0 * gate.float().mean().item()

        clean_metrics, clean_gate_rate = evaluate_features(
            clean_logits, clean_refs, clean_stability, clean_ranks
        )
        top1_metrics, top1_gate_rate = evaluate_features(
            top1_logits, top1_refs, top1_stability, top1_ranks
        )
        set2_metrics, set2_gate_rate = evaluate_features(
            set2_logits, set2_refs, set2_stability, set2_ranks
        )
        feasible = (
            clean_metrics["set2"] >= target_clean_set2
            and clean_metrics["top1"] >= top1_floor
        )
        proxy_score = top1_metrics["top1"] + set2_metrics["set2"]
        rows.append(
            {
                "stability_threshold": float(stability_threshold),
                "support_topk": int(support_topk),
                "set2_logit_bias": float(bias),
                "clean_top1": clean_metrics["top1"],
                "clean_set2": clean_metrics["set2"],
                "clean_gate_rate": clean_gate_rate,
                "proxy_top1": top1_metrics["top1"],
                "proxy_top1_set2": top1_metrics["set2"],
                "top1_proxy_gate_rate": top1_gate_rate,
                "proxy_set2_top1": set2_metrics["top1"],
                "proxy_set2": set2_metrics["set2"],
                "set2_proxy_gate_rate": set2_gate_rate,
                "proxy_score": proxy_score,
                "target_clean_set2": target_clean_set2,
                "clean_top1_floor": top1_floor,
                "feasible": int(feasible),
                "selected": 0,
            }
        )

    feasible_rows = [row for row in rows if row["feasible"] == 1]
    if not feasible_rows:
        best_clean = max(
            rows,
            key=lambda row: (
                float(row["clean_set2"]), float(row["clean_top1"])
            ),
        )
        raise RuntimeError(
            "No gated configuration met the clean constraints. "
            f"Best clean point: top1={best_clean['clean_top1']:.2f}, "
            f"set2={best_clean['clean_set2']:.2f}. Expand the gate grids or "
            "lower --gated-target-clean-set2 without tuning on test."
        )

    selected = max(
        feasible_rows,
        key=lambda row: (
            float(row["proxy_score"]),
            float(row["proxy_set2"]),
            float(row["proxy_top1"]),
            -float(row["set2_logit_bias"]),
        ),
    )
    selected["selected"] = 1
    parameters = {
        "gate_stability_threshold": float(selected["stability_threshold"]),
        "gate_support_topk": float(selected["support_topk"]),
        "set2_logit_bias": float(selected["set2_logit_bias"]),
        "clean_top1": float(selected["clean_top1"]),
        "clean_set2": float(selected["clean_set2"]),
        "proxy_top1": float(selected["proxy_top1"]),
        "proxy_set2": float(selected["proxy_set2"]),
    }
    return parameters, rows, clean_baseline


def calibrate_proposed_sigma(
    standard_model: nn.Module,
    robust_model: nn.Module,
    images: torch.Tensor,
    reference_logits: torch.Tensor,
    alpha: float,
    targeted_epsilon: float,
    targeted_steps: int,
    top1_weight: float,
    set2_weight: float,
    target_drop: float,
    sigma_min: float,
    sigma_max: float,
    iterations: int,
    trials: int,
    tolerance: float,
    batch_size: int,
) -> Tuple[float, float, float]:
    sigma_zero = CrossModelTop2Defense(
        standard_model=standard_model,
        robust_model=robust_model,
        alpha=alpha,
        epsilon=targeted_epsilon,
        steps=targeted_steps,
        sigma=0.0,
        top1_weight=top1_weight,
        set2_weight=set2_weight,
    )
    baseline_set2 = mean_clean_set2(
        sigma_zero, images, reference_logits, batch_size, trials=1
    )
    target_value = baseline_set2 - target_drop
    low, high = sigma_min, sigma_max
    best_sigma, best_value = sigma_min, baseline_set2

    for _ in range(iterations):
        middle = (low + high) / 2.0
        defense = CrossModelTop2Defense(
            standard_model=standard_model,
            robust_model=robust_model,
            alpha=alpha,
            epsilon=targeted_epsilon,
            steps=targeted_steps,
            sigma=middle,
            top1_weight=top1_weight,
            set2_weight=set2_weight,
        )
        value = mean_clean_set2(
            defense, images, reference_logits, batch_size, trials=trials
        )
        if value >= target_value - tolerance:
            best_sigma, best_value = middle, value
            low = middle
        else:
            high = middle

    return best_sigma, best_value, baseline_set2


def top1_attack_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Positive while the true label is Top-1; minimize to break Top-1."""
    row = torch.arange(len(labels), device=labels.device)
    correct = logits[row, labels]
    other_logits = logits.clone()
    other_logits[row, labels] = -1e9
    return correct - other_logits.max(dim=1).values


def set2_attack_margin(
    logits: torch.Tensor, reference_top2: torch.Tensor
) -> torch.Tensor:
    """Positive while the clean reference Top-2 set is preserved."""
    reference_values = logits.gather(1, reference_top2)
    weakest_reference = reference_values.min(dim=1).values
    outside_mask = torch.ones_like(logits, dtype=torch.bool)
    outside_mask.scatter_(1, reference_top2, False)
    strongest_outside = logits.masked_fill(
        ~outside_mask, float("-inf")
    ).max(dim=1).values
    return weakest_reference - strongest_outside


def square_attack(
    query_fn: Callable,
    x: torch.Tensor,
    score_fn: Callable,
    eps: float,
    n_queries: int,
) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x_best = x.detach().clone()
    best_margin = score_fn(query_fn(x_best))
    if n_queries > 1:
        random_candidate = torch.clamp(
            x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0
        )
        random_margin = score_fn(query_fn(random_candidate))
        improved = random_margin < best_margin
        x_best[improved] = random_candidate[improved]
        best_margin[improved] = random_margin[improved]

    for query_index in range(2, n_queries):
        ratio = max(1.0 - query_index / n_queries, 0.05)
        patch_size = max(int(ratio * height), 1)
        rows = np.random.randint(0, max(height - patch_size + 1, 1), size=batch)
        cols = np.random.randint(0, max(width - patch_size + 1, 1), size=batch)
        candidate = x_best.clone()
        for sample in range(batch):
            if best_margin[sample] <= 0:
                continue
            new_value = (
                2
                * torch.randint(
                    0,
                    2,
                    (channels, patch_size, patch_size),
                    device=x.device,
                ).float()
                - 1
            ) * eps
            patch = (
                x[sample, :, rows[sample] : rows[sample] + patch_size,
                  cols[sample] : cols[sample] + patch_size]
                + new_value
            )
            candidate[
                sample,
                :,
                rows[sample] : rows[sample] + patch_size,
                cols[sample] : cols[sample] + patch_size,
            ] = patch
            candidate[sample] = torch.clamp(
                torch.clamp(candidate[sample], x[sample] - eps, x[sample] + eps),
                0.0,
                1.0,
            )
        candidate_margin = score_fn(query_fn(candidate))
        improved = candidate_margin < best_margin
        x_best[improved] = candidate[improved]
        best_margin[improved] = candidate_margin[improved]
    return x_best.detach()


def nes_attack(
    query_fn: Callable,
    x: torch.Tensor,
    score_fn: Callable,
    eps: float,
    n_queries: int,
    sigma: float = 0.001,
    learning_rate: float = 0.01,
    n_samples: int = 50,
) -> torch.Tensor:
    adversarial = x.clone()
    steps = n_queries // n_samples
    for _ in range(steps):
        gradient_estimate = torch.zeros_like(adversarial)
        for _ in range(n_samples // 2):
            noise = torch.randn_like(adversarial)
            loss_plus = -score_fn(
                query_fn(torch.clamp(adversarial + sigma * noise, 0.0, 1.0))
            )
            loss_minus = -score_fn(
                query_fn(torch.clamp(adversarial - sigma * noise, 0.0, 1.0))
            )
            gradient_estimate += (
                (loss_plus - loss_minus).view(-1, 1, 1, 1) * noise
            )
        gradient_estimate /= n_samples * sigma
        adversarial = torch.clamp(
            adversarial + learning_rate * gradient_estimate.sign(),
            x - eps,
            x + eps,
        ).clamp(0.0, 1.0)
    return adversarial.detach()


def signhunt_attack(
    query_fn: Callable,
    x: torch.Tensor,
    score_fn: Callable,
    eps: float,
    n_queries: int,
) -> torch.Tensor:
    batch, channels, height, width = x.shape
    dimension = channels * height * width
    adversarial = x.detach().clone()
    best_margin = score_fn(query_fn(adversarial))
    signs = 2 * torch.randint(0, 2, x.shape, device=x.device).float() - 1
    if n_queries > 1:
        random_candidate = torch.clamp(x + eps * signs, 0.0, 1.0)
        random_candidate = torch.clamp(random_candidate, x - eps, x + eps)
        random_margin = score_fn(query_fn(random_candidate))
        improved = random_margin < best_margin
        adversarial[improved] = random_candidate[improved]
        best_margin[improved] = random_margin[improved]

    for query_index in range(2, n_queries):
        decay = max(1.0 - query_index / n_queries, 0.02)
        number_to_flip = max(1, int(dimension * 0.1 * decay))
        new_signs = signs.clone()
        for sample in range(batch):
            if best_margin[sample] <= 0:
                continue
            indices = torch.randperm(dimension, device=x.device)[:number_to_flip]
            new_signs[sample].view(-1)[indices] *= -1
        candidate = torch.clamp(
            torch.clamp(x + eps * new_signs, 0.0, 1.0), x - eps, x + eps
        )
        candidate_margin = score_fn(query_fn(candidate))
        improved = candidate_margin < best_margin
        adversarial[improved] = candidate[improved]
        best_margin[improved] = candidate_margin[improved]
        signs[improved] = new_signs[improved]
    return adversarial.detach()


ATTACKS: Dict[str, Callable] = {
    "Square": square_attack,
    "NES": nes_attack,
    "SignHunt": signhunt_attack,
}


def evaluate_under_attack(
    defense: Callable,
    attack: Callable,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    epsilon: float,
    n_queries: int,
    batch_size: int,
    attack_eot: int,
    eval_mc: int,
    attack_goal: str,
) -> Dict[str, float]:
    adversarial_logits = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    for batch_number, low in enumerate(range(0, len(images), batch_size), start=1):
        high = min(low + batch_size, len(images))
        image_batch = images[low:high]
        label_batch = labels[low:high]
        reference_top2_batch = reference_logits[low:high].to(
            image_batch.device
        ).topk(2, dim=1).indices

        def query_fn(candidate: torch.Tensor) -> torch.Tensor:
            return averaged_query(defense, candidate, attack_eot)

        if attack_goal == "top1":
            def score_fn(logits: torch.Tensor) -> torch.Tensor:
                return top1_attack_margin(logits, label_batch)
        elif attack_goal == "set2":
            def score_fn(logits: torch.Tensor) -> torch.Tensor:
                return set2_attack_margin(logits, reference_top2_batch)
        else:
            raise ValueError(f"Unknown attack goal: {attack_goal}")

        attacked = attack(
            query_fn,
            image_batch,
            score_fn,
            eps=epsilon,
            n_queries=n_queries,
        )
        adversarial_logits.append(
            averaged_query(defense, attacked, eval_mc).detach().cpu()
        )
        print(
            f"    batch {batch_number}/{n_batches} complete",
            flush=True,
        )
    logits = torch.cat(adversarial_logits, dim=0)
    return all_metrics(logits, labels.cpu(), reference_logits.cpu())


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_completed(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (
                row["defense"],
                row["attack"],
                row["attack_goal"],
                int(row["budget"]),
            )
            for row in csv.DictReader(handle)
        }


def make_wide_table(long_path: Path, wide_path: Path) -> None:
    if not long_path.exists():
        return
    with long_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    defenses = list(dict.fromkeys(row["defense"] for row in rows))
    columns = list(
        dict.fromkeys(
            (row["attack"], row["attack_goal"], row["budget"])
            for row in rows
        )
    )
    lookup = {
        (
            row["defense"], row["attack"], row["attack_goal"], row["budget"]
        ): row
        for row in rows
    }
    fieldnames = ["defense"] + [
        f"{attack}_{goal}_q{budget}" for attack, goal, budget in columns
    ]
    output_rows = []
    for defense in defenses:
        output: Dict[str, object] = {"defense": defense}
        for attack, goal, budget in columns:
            row = lookup.get((defense, attack, goal, budget))
            column_name = f"{attack}_{goal}_q{budget}"
            if row is None:
                output[column_name] = ""
            else:
                output[column_name] = (
                    f"{float(row['top1']):.1f} / "
                    f"{float(row['set2']):.1f}"
                )
        output_rows.append(output)
    write_rows(wide_path, output_rows)


def sigma_drop_tag(drop: float) -> str:
    text = f"{float(drop):g}"
    return text.replace(".", "p")


def selected_defense_names(
    args: argparse.Namespace, available: Sequence[str]
) -> List[str]:
    requested = args.defenses
    if requested == ["all"]:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown --defenses entries: {unknown}; valid={list(available)}"
        )
    return requested


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_examples = 100
        args.calibration_examples = min(args.calibration_examples, 100)
        args.attacks = ["Square", "SignHunt"]
        args.budgets = [100, 500]
        args.sigma_search_iters = min(args.sigma_search_iters, 10)
        args.sigma_trials = min(args.sigma_trials, 2)
        args.strength_alpha_grid = args.strength_alpha_grid[:2]
        args.strength_step_grid = args.strength_step_grid[:1]
        args.top1_weight_grid = args.top1_weight_grid[:2]
        args.set2_weight_grid = args.set2_weight_grid[:2]
        args.logit_bias_grid_size = min(args.logit_bias_grid_size, 41)
        args.logit_bias_calibration_mc = min(args.logit_bias_calibration_mc, 2)
        args.gate_calibration_mc = min(args.gate_calibration_mc, 2)
        args.proxy_pgd_steps = min(args.proxy_pgd_steps, 5)
    if args.attack_eot < 1 or args.eval_mc < 1 or args.sigma_trials < 1:
        raise ValueError("attack-eot, eval-mc, and sigma-trials must be >= 1")
    if args.top1_weight < 0.0:
        raise ValueError("--top1-weight must be non-negative")
    if args.set2_weight < 0.0:
        raise ValueError("--set2-weight must be non-negative")
    if args.strength_steps < 1 or any(x < 1 for x in args.strength_step_grid):
        raise ValueError("All strengthening step counts must be >= 1")
    if args.strength_alpha <= 0.0 or any(x <= 0.0 for x in args.strength_alpha_grid):
        raise ValueError("All strengthening alpha values must be positive")
    if any(x < 0.0 for x in args.top1_weight_grid + args.set2_weight_grid):
        raise ValueError("All objective weights must be non-negative")
    if args.tune_max_top1_drop < 0.0:
        raise ValueError("--tune-max-top1-drop must be non-negative")
    if not 0.0 < args.target_clean_set2 <= 100.0:
        raise ValueError("--target-clean-set2 must be in (0, 100]")
    if args.logit_bias_max <= 0.0 or args.logit_bias_grid_size < 2:
        raise ValueError("logit-bias-max must be positive and grid-size >= 2")
    if args.logit_bias_calibration_mc < 1:
        raise ValueError("--logit-bias-calibration-mc must be >= 1")
    if args.logit_bias_max_top1_drop < 0.0:
        raise ValueError("--logit-bias-max-top1-drop must be non-negative")
    if not 0.0 < args.gated_target_clean_set2 <= 100.0:
        raise ValueError("--gated-target-clean-set2 must be in (0, 100]")
    if args.gate_views < 0 or args.gate_view_sigma < 0.0:
        raise ValueError("gate-views and gate-view-sigma must be non-negative")
    if args.gate_calibration_mc < 1 or args.proxy_pgd_steps < 1:
        raise ValueError("gate-calibration-mc and proxy-pgd-steps must be >= 1")
    if any(not 0.0 <= x <= 1.0 for x in args.gate_stability_grid):
        raise ValueError("gate-stability-grid values must be in [0, 1]")
    if any(x < 2 or x > 10 for x in args.gate_support_topk_grid):
        raise ValueError("gate-support-topk-grid values must be between 2 and 10")
    if any(x < 0.0 for x in args.gate_bias_grid):
        raise ValueError("gate-bias-grid values must be non-negative")
    if args.gate_max_clean_top1_drop < 0.0:
        raise ValueError("--gate-max-clean-top1-drop must be non-negative")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but no GPU is available")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    standard_model, standard_arch = load_resnet20(
        args.standard_weights, device, STANDARD_CIFAR10_STD
    )
    robust_model, robust_arch = load_resnet20(
        args.robust_weights, device, ROBUST_CIFAR10_STD
    )
    print(f"Standard checkpoint: {args.standard_weights} ({standard_arch})")
    print(f"Robust checkpoint:   {args.robust_weights} ({robust_arch})")

    images, labels = load_balanced_subset(
        args.data_dir, args.n_examples, args.num_workers, device, args.seed
    )
    calibration_images, calibration_labels = load_calibration_subset(
        data_dir=args.data_dir,
        n_examples=args.calibration_examples,
        training_val_size=args.training_val_size,
        split_seed=args.training_split_seed,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"Balanced test subset: {tuple(images.shape)}")
    print("Class counts:", torch.bincount(labels).tolist())
    print(
        "Held-out calibration subset:", tuple(calibration_images.shape),
        "(never used for reported test metrics)"
    )

    standard_defense = NoDefense(standard_model)
    robust_defense = NoDefense(robust_model)
    reference_logits = compute_logits_batched(
        standard_defense, images, args.batch_size, mc_trials=1
    )
    calibration_reference_logits = compute_logits_batched(
        standard_defense, calibration_images, args.batch_size, mc_trials=1
    )

    if args.no_tune_strength:
        alpha_values = [args.strength_alpha]
        step_values = [args.strength_steps]
        top1_weight_values = [args.top1_weight]
        set2_weight_values = [args.set2_weight]
    else:
        alpha_values = args.strength_alpha_grid
        step_values = args.strength_step_grid
        top1_weight_values = args.top1_weight_grid
        set2_weight_values = args.set2_weight_grid
    selected, strength_rows, robust_calibration_metrics = tune_strength_parameters(
        standard_model=standard_model,
        robust_model=robust_model,
        images=calibration_images,
        labels=calibration_labels,
        reference_logits=calibration_reference_logits,
        batch_size=args.batch_size,
        epsilon=args.strength_epsilon,
        alpha_values=alpha_values,
        step_values=step_values,
        top1_weight_values=top1_weight_values,
        set2_weight_values=set2_weight_values,
        max_top1_drop=args.tune_max_top1_drop,
    )
    args.strength_alpha = float(selected["alpha"])
    args.strength_steps = int(selected["steps"])
    args.top1_weight = float(selected["top1_weight"])
    args.set2_weight = float(selected["set2_weight"])
    write_rows(output_dir / "strength_calibration.csv", strength_rows)

    targeted_defense = CrossModelTop2Defense(
        standard_model=standard_model,
        robust_model=robust_model,
        alpha=args.strength_alpha,
        epsilon=args.strength_epsilon,
        steps=args.strength_steps,
        sigma=0.0,
        top1_weight=args.top1_weight,
        set2_weight=args.set2_weight,
    )
    print(
        "Selected targeted defense: "
        f"alpha={args.strength_alpha:g}, steps={args.strength_steps}, "
        f"epsilon={args.strength_epsilon:g}"
    )
    print(
        "Exact objective: "
        f"{args.top1_weight:g}*[z(c1)-max z(not c1)] + "
        f"{args.set2_weight:g}*[min z(c1,c2)-max z(outside)]"
    )

    sigma_by_drop: Dict[float, float] = {}
    calibration_rows = []
    for drop in args.sigma_drops:
        sigma, achieved, targeted_baseline = calibrate_proposed_sigma(
            standard_model=standard_model,
            robust_model=robust_model,
            images=calibration_images,
            reference_logits=calibration_reference_logits,
            alpha=args.strength_alpha,
            targeted_epsilon=args.strength_epsilon,
            targeted_steps=args.strength_steps,
            top1_weight=args.top1_weight,
            set2_weight=args.set2_weight,
            target_drop=drop,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            iterations=args.sigma_search_iters,
            trials=args.sigma_trials,
            tolerance=args.sigma_tolerance,
            batch_size=args.batch_size,
        )
        sigma_by_drop[float(drop)] = sigma
        calibration_rows.append(
            {
                "target_set2_drop_pp": drop,
                "top1_weight": args.top1_weight,
                "set2_weight": args.set2_weight,
                "strength_alpha": args.strength_alpha,
                "strength_steps": args.strength_steps,
                "sigma": sigma,
                "targeted_sigma0_set2": targeted_baseline,
                "achieved_set2": achieved,
                "achieved_drop_pp": targeted_baseline - achieved,
            }
        )
        print(
            f"Calibrated drop={drop:g}pp -> sigma={sigma:.7f}; "
            f"set2 {targeted_baseline:.2f} -> {achieved:.2f}"
        )
    write_rows(output_dir / "sigma_calibration.csv", calibration_rows)
    with (output_dir / "calibrated_sigmas.json").open("w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in sigma_by_drop.items()}, handle, indent=2)

    # The same calibrated sigma is used for Gaussian-only and proposed rows,
    # so the targeted component is compared under identical random-noise scale.
    defenses: Dict[str, Callable] = {
        "standard": standard_defense,
        "robust": robust_defense,
        "targeted": targeted_defense,
    }
    defense_sigmas: Dict[str, float] = {
        "standard": 0.0,
        "robust": 0.0,
        "targeted": 0.0,
    }
    defense_biases: Dict[str, Tuple[float, float]] = {
        "standard": (0.0, 0.0),
        "robust": (0.0, 0.0),
        "targeted": (0.0, 0.0),
    }
    defense_gates: Dict[str, Tuple[float, int]] = {
        "standard": (0.0, 10),
        "robust": (0.0, 10),
        "targeted": (0.0, 10),
    }
    for drop, sigma in sigma_by_drop.items():
        tag = sigma_drop_tag(drop)
        gaussian_name = f"gaussian{tag}"
        proposed_name = f"proposed{tag}"
        defenses[gaussian_name] = GaussianInputNoise(robust_model, sigma)
        defenses[proposed_name] = CrossModelTop2Defense(
            standard_model=standard_model,
            robust_model=robust_model,
            alpha=args.strength_alpha,
            epsilon=args.strength_epsilon,
            steps=args.strength_steps,
            sigma=sigma,
            top1_weight=args.top1_weight,
            set2_weight=args.set2_weight,
        )
        defense_sigmas[gaussian_name] = sigma
        defense_sigmas[proposed_name] = sigma
        defense_biases[gaussian_name] = (0.0, 0.0)
        defense_biases[proposed_name] = (0.0, 0.0)
        defense_gates[gaussian_name] = (0.0, 10)
        defense_gates[proposed_name] = (0.0, 10)

    bias_parameters = {
        "set2_logit_bias": 0.0,
        "top1_logit_bias": 0.0,
        "validation_top1": 0.0,
        "validation_set2": 0.0,
    }
    bias_baseline = {"top1": 0.0, "set2": 0.0}
    alignment_requested = args.defenses == ["all"] or "proposed80" in args.defenses
    if alignment_requested:
        matched_drops = [
            drop for drop in sigma_by_drop
            if abs(drop - args.alignment_sigma_drop) < 1e-9
        ]
        if not matched_drops:
            raise ValueError(
                "proposed80 requires --alignment-sigma-drop to also appear in "
                "--sigma-drops"
            )
        alignment_drop = matched_drops[0]
        alignment_sigma = sigma_by_drop[alignment_drop]
        unaligned_for_bias = CrossModelTop2Defense(
            standard_model=standard_model,
            robust_model=robust_model,
            alpha=args.strength_alpha,
            epsilon=args.strength_epsilon,
            steps=args.strength_steps,
            sigma=alignment_sigma,
            top1_weight=args.top1_weight,
            set2_weight=args.set2_weight,
        )
        calibration_unaligned_logits = compute_logits_batched(
            unaligned_for_bias,
            calibration_images,
            args.batch_size,
            mc_trials=args.logit_bias_calibration_mc,
        )
        bias_parameters, bias_rows, bias_baseline = tune_reference_logit_bias(
            base_logits=calibration_unaligned_logits,
            labels=calibration_labels,
            reference_logits=calibration_reference_logits,
            target_set2=args.target_clean_set2,
            max_top1_drop=args.logit_bias_max_top1_drop,
            bias_max=args.logit_bias_max,
            grid_size=args.logit_bias_grid_size,
        )
        write_rows(output_dir / "logit_bias_calibration.csv", bias_rows)
        defenses["proposed80"] = CrossModelTop2Defense(
            standard_model=standard_model,
            robust_model=robust_model,
            alpha=args.strength_alpha,
            epsilon=args.strength_epsilon,
            steps=args.strength_steps,
            sigma=alignment_sigma,
            top1_weight=args.top1_weight,
            set2_weight=args.set2_weight,
            set2_logit_bias=bias_parameters["set2_logit_bias"],
            top1_logit_bias=bias_parameters["top1_logit_bias"],
        )
        defense_sigmas["proposed80"] = alignment_sigma
        defense_biases["proposed80"] = (
            bias_parameters["set2_logit_bias"],
            bias_parameters["top1_logit_bias"],
        )
        defense_gates["proposed80"] = (0.0, 10)
        print(
            "Calibrated proposed80: "
            f"set2_bias={bias_parameters['set2_logit_bias']:.4f}, "
            f"top1_bias={bias_parameters['top1_logit_bias']:.4f}; "
            f"validation top1 {bias_baseline['top1']:.2f} -> "
            f"{bias_parameters['validation_top1']:.2f}, set2 "
            f"{bias_baseline['set2']:.2f} -> "
            f"{bias_parameters['validation_set2']:.2f}"
        )

    gate_parameters = {
        "gate_stability_threshold": 0.0,
        "gate_support_topk": 10.0,
        "set2_logit_bias": 0.0,
        "clean_top1": 0.0,
        "clean_set2": 0.0,
        "proxy_top1": 0.0,
        "proxy_set2": 0.0,
    }
    gate_baseline = {"top1": 0.0, "set2": 0.0}
    gate_requested = args.defenses == ["all"] or "proposed_gated" in args.defenses
    if gate_requested:
        matched_drops = [
            drop for drop in sigma_by_drop
            if abs(drop - args.alignment_sigma_drop) < 1e-9
        ]
        if not matched_drops:
            raise ValueError(
                "proposed_gated requires --alignment-sigma-drop to also appear "
                "in --sigma-drops"
            )
        gate_sigma = sigma_by_drop[matched_drops[0]]
        clean_reference_top2_device = calibration_reference_logits.to(
            device
        ).topk(2, dim=1).indices
        print("Generating validation PGD proxies for adaptive-gate tuning...")
        top1_proxy_images = pgd_proxy_attack(
            model=robust_model,
            images=calibration_images,
            labels=calibration_labels,
            clean_reference_top2=clean_reference_top2_device,
            goal="top1",
            epsilon=args.attack_epsilon,
            alpha=args.proxy_pgd_alpha,
            steps=args.proxy_pgd_steps,
        )
        set2_proxy_images = pgd_proxy_attack(
            model=standard_model,
            images=calibration_images,
            labels=calibration_labels,
            clean_reference_top2=clean_reference_top2_device,
            goal="set2",
            epsilon=args.attack_epsilon,
            alpha=args.proxy_pgd_alpha,
            steps=args.proxy_pgd_steps,
        )
        gate_feature_defense = CrossModelTop2Defense(
            standard_model=standard_model,
            robust_model=robust_model,
            alpha=args.strength_alpha,
            epsilon=args.strength_epsilon,
            steps=args.strength_steps,
            sigma=gate_sigma,
            top1_weight=args.top1_weight,
            set2_weight=args.set2_weight,
            gate_enabled=True,
            gate_views=args.gate_views,
            gate_view_sigma=args.gate_view_sigma,
        )
        clean_gate_features = compute_gate_features_batched(
            gate_feature_defense,
            calibration_images,
            args.batch_size,
            args.gate_calibration_mc,
        )
        top1_proxy_gate_features = compute_gate_features_batched(
            gate_feature_defense,
            top1_proxy_images,
            args.batch_size,
            args.gate_calibration_mc,
        )
        set2_proxy_gate_features = compute_gate_features_batched(
            gate_feature_defense,
            set2_proxy_images,
            args.batch_size,
            args.gate_calibration_mc,
        )
        gate_parameters, gate_rows, gate_baseline = tune_gated_alignment(
            clean_features=clean_gate_features,
            top1_proxy_features=top1_proxy_gate_features,
            set2_proxy_features=set2_proxy_gate_features,
            labels=calibration_labels,
            clean_reference_logits=calibration_reference_logits,
            target_clean_set2=args.gated_target_clean_set2,
            max_clean_top1_drop=args.gate_max_clean_top1_drop,
            stability_values=args.gate_stability_grid,
            support_topk_values=args.gate_support_topk_grid,
            bias_values=args.gate_bias_grid,
        )
        write_rows(output_dir / "gate_calibration.csv", gate_rows)
        defenses["proposed_gated"] = CrossModelTop2Defense(
            standard_model=standard_model,
            robust_model=robust_model,
            alpha=args.strength_alpha,
            epsilon=args.strength_epsilon,
            steps=args.strength_steps,
            sigma=gate_sigma,
            top1_weight=args.top1_weight,
            set2_weight=args.set2_weight,
            set2_logit_bias=gate_parameters["set2_logit_bias"],
            top1_logit_bias=0.0,
            gate_enabled=True,
            gate_views=args.gate_views,
            gate_view_sigma=args.gate_view_sigma,
            gate_stability_threshold=gate_parameters[
                "gate_stability_threshold"
            ],
            gate_support_topk=int(gate_parameters["gate_support_topk"]),
        )
        defense_sigmas["proposed_gated"] = gate_sigma
        defense_biases["proposed_gated"] = (
            gate_parameters["set2_logit_bias"], 0.0
        )
        defense_gates["proposed_gated"] = (
            gate_parameters["gate_stability_threshold"],
            int(gate_parameters["gate_support_topk"]),
        )
        print(
            "Calibrated proposed_gated: "
            f"bias={gate_parameters['set2_logit_bias']:.3f}, "
            f"stability>={gate_parameters['gate_stability_threshold']:.3f}, "
            f"support_topk={int(gate_parameters['gate_support_topk'])}; "
            f"clean top1/set2={gate_parameters['clean_top1']:.2f}/"
            f"{gate_parameters['clean_set2']:.2f}; proxy top1/set2="
            f"{gate_parameters['proxy_top1']:.2f}/"
            f"{gate_parameters['proxy_set2']:.2f}"
        )

    chosen_names = selected_defense_names(args, list(defenses))
    defenses = {name: defenses[name] for name in chosen_names}

    selected_configuration = {
        "robust_checkpoint": str(args.robust_weights.expanduser().resolve()),
        "strength_alpha": args.strength_alpha,
        "strength_steps": args.strength_steps,
        "strength_epsilon": args.strength_epsilon,
        "top1_weight": args.top1_weight,
        "set2_weight": args.set2_weight,
        "calibration_robust_top1": robust_calibration_metrics["top1"],
        "calibration_robust_set2": robust_calibration_metrics["set2"],
        "sigma_by_set2_drop": {str(k): v for k, v in sigma_by_drop.items()},
        "target_clean_set2": args.target_clean_set2,
        "alignment_sigma_drop": args.alignment_sigma_drop,
        "set2_logit_bias": bias_parameters["set2_logit_bias"],
        "top1_logit_bias": bias_parameters["top1_logit_bias"],
        "alignment_validation_baseline": bias_baseline,
        "alignment_validation_top1": bias_parameters["validation_top1"],
        "alignment_validation_set2": bias_parameters["validation_set2"],
        "gated_target_clean_set2": args.gated_target_clean_set2,
        "gate_views": args.gate_views,
        "gate_view_sigma": args.gate_view_sigma,
        "gate_stability_threshold": gate_parameters[
            "gate_stability_threshold"
        ],
        "gate_support_topk": gate_parameters["gate_support_topk"],
        "gate_set2_logit_bias": gate_parameters["set2_logit_bias"],
        "gate_validation_baseline": gate_baseline,
        "gate_validation_clean_top1": gate_parameters["clean_top1"],
        "gate_validation_clean_set2": gate_parameters["clean_set2"],
        "gate_validation_proxy_top1": gate_parameters["proxy_top1"],
        "gate_validation_proxy_set2": gate_parameters["proxy_set2"],
    }
    with (output_dir / "selected_defense_parameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(selected_configuration, handle, indent=2)

    clean_rows = []
    for defense_name, defense in defenses.items():
        clean_logits = compute_logits_batched(
            defense, images, args.batch_size, mc_trials=args.eval_mc
        )
        metrics = all_metrics(clean_logits, labels, reference_logits)
        row: Dict[str, object] = {
            "defense": defense_name,
            "strength_alpha": args.strength_alpha,
            "strength_steps": args.strength_steps,
            "top1_weight": args.top1_weight,
            "set2_weight": args.set2_weight,
            "sigma": defense_sigmas[defense_name],
            "set2_logit_bias": defense_biases[defense_name][0],
            "top1_logit_bias": defense_biases[defense_name][1],
            "gate_stability_threshold": defense_gates[defense_name][0],
            "gate_support_topk": defense_gates[defense_name][1],
            **metrics,
        }
        clean_rows.append(row)
        print(
            f"CLEAN {defense_name:10s} | "
            f"top1={metrics['top1']:.2f} set2={metrics['set2']:.2f}"
        )
        if defense_name == "proposed80" and metrics["set2"] < 80.0:
            print(
                "WARNING: proposed80 validation-calibrated successfully, but "
                f"this independent test subset reached only {metrics['set2']:.2f}% "
                "Set-2. Do not tune on test; increase the validation target and "
                "rerun in a new output directory if a larger buffer is needed."
            )
        if defense_name == "proposed_gated" and metrics["set2"] < 80.0:
            print(
                "WARNING: proposed_gated did not generalize above 80% clean "
                f"Set-2 on this independent subset ({metrics['set2']:.2f}%). "
                "Do not tune gate parameters on test."
            )
    write_rows(output_dir / "clean_results.csv", clean_rows)

    if args.clean_only:
        print("Clean-only evaluation complete. Results directory:", output_dir)
        return

    results_path = output_dir / "attack_results.csv"
    completed = read_completed(results_path) if args.resume_results else set()
    if not args.resume_results and results_path.exists():
        results_path.unlink()

    for defense_name, defense in defenses.items():
        for attack_name in args.attacks:
            attack = ATTACKS[attack_name]
            for attack_goal in args.attack_goals:
                for budget in args.budgets:
                    key = (defense_name, attack_name, attack_goal, budget)
                    if key in completed:
                        print(f"SKIP completed: {key}")
                        continue
                    print(
                        f"EVALUATING defense={defense_name} attack={attack_name} "
                        f"goal={attack_goal} queries={budget}",
                        flush=True,
                    )
                    metrics = evaluate_under_attack(
                        defense=defense,
                        attack=attack,
                        images=images,
                        labels=labels,
                        reference_logits=reference_logits,
                        epsilon=args.attack_epsilon,
                        n_queries=budget,
                        batch_size=args.batch_size,
                        attack_eot=args.attack_eot,
                        eval_mc=args.eval_mc,
                        attack_goal=attack_goal,
                    )
                    row = {
                        "defense": defense_name,
                        "attack": attack_name,
                        "attack_goal": attack_goal,
                        "budget": budget,
                        "strength_alpha": args.strength_alpha,
                        "strength_steps": args.strength_steps,
                        "top1_weight": args.top1_weight,
                        "set2_weight": args.set2_weight,
                        "sigma": defense_sigmas[defense_name],
                        "set2_logit_bias": defense_biases[defense_name][0],
                        "top1_logit_bias": defense_biases[defense_name][1],
                        "gate_stability_threshold": defense_gates[
                            defense_name
                        ][0],
                        "gate_support_topk": defense_gates[defense_name][1],
                        "attack_epsilon": args.attack_epsilon,
                        "attack_eot": args.attack_eot,
                        "eval_mc": args.eval_mc,
                        **metrics,
                    }
                    append_row(results_path, row)
                    make_wide_table(
                        results_path, output_dir / "comparison_table.csv"
                    )
                    print(
                        f"RESULT {defense_name}/{attack_name}/"
                        f"goal={attack_goal}/q={budget}: "
                        f"top1={metrics['top1']:.2f} "
                        f"set2={metrics['set2']:.2f}",
                        flush=True,
                    )

    print("Evaluation complete. Results directory:", output_dir)
    print("Cell format in comparison_table.csv: top1 / set2")


if __name__ == "__main__":
    main()
