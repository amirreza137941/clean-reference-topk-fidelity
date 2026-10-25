#!/usr/bin/env python3
"""Leakage-free dual-backbone CIFAR-100 defense for protocol R2.

The defense contains two independent ResNet-20 branches:

* ``robust_branch`` is initialized from the strongest available adversarial
  checkpoint, remains frozen, and determines final Top-1.
* ``set2_branch`` is initialized from the high-clean standard checkpoint and
  predicts the two-class set.  Only its late layers are adversarially fine-
  tuned by default.

Final logits always use the robust branch's primary class and the Set2 branch's
best remaining class.  The hard primary selector has a straight-through soft
gradient so PGD can attack the actual composed defense instead of encountering
a zero-gradient argmax.

The standard teacher is a frozen reference during training only.  At inference
the deployed defense consists of the robust and trained Set2 branches; a third
teacher model is not required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import topk_cifar100_r2_common as protocol


CIFAR100_MEAN = protocol.CIFAR100_MEAN
CIFAR100_STD = protocol.CIFAR100_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dual-backbone robust-Top1/teacher-Set2 defense"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=protocol.DEFAULT_DATA,
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=(
            protocol.DEFAULT_ROOT
            / "standard/resnet20_standard_best_validation_weights.pt"
        ),
    )
    parser.add_argument(
        "--robust-anchor-weights",
        type=Path,
        default=(
            protocol.DEFAULT_ROOT
            / "teacher_top2_at/resnet20_teacher_top2_best_composite_weights.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=protocol.DEFAULT_ROOT / "dual_top2",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=protocol.DEFAULT_ROOT / "frozen_split.json",
        help="Immutable split created by the R2 preflight",
    )

    parser.add_argument(
        "--trainable-set2-layers",
        choices=["fc", "layer3_fc", "all"],
        default="layer3_fc",
        help="How much of the standard-initialized Set2 branch is trainable",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--selector-temperature", type=float, default=1.0)
    parser.add_argument("--primary-margin", type=float, default=1.0)
    parser.add_argument("--set2-margin", type=float, default=0.50)
    parser.add_argument("--kd-temperature", type=float, default=4.0)

    # Branch objective.  Clean KD and hard ranking retain standard-model Set2;
    # adversarial KD/ranking try to make that set stable around the input.
    parser.add_argument("--bce-weight", type=float, default=0.20)
    parser.add_argument("--raw-ranking-weight", type=float, default=1.00)
    parser.add_argument("--composed-ranking-weight", type=float, default=0.50)
    parser.add_argument("--kd-weight", type=float, default=1.00)
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=49.0,
        help="Class-balanced BCE weight: (100-2)/2 = 49",
    )
    parser.add_argument("--clean-loss-weight", type=float, default=0.60)
    parser.add_argument("--adversarial-loss-weight", type=float, default=0.40)

    # Constraint multiplier on the clean branch loss.
    parser.add_argument(
        "--clean-set2-constraint",
        type=float,
        default=None,
        help=(
            "Frozen clean-validation Exact-Set2 floor. By default it is "
            "initial Exact-Set2 minus --clean-set2-max-drop."
        ),
    )
    parser.add_argument("--clean-set2-max-drop", type=float, default=2.0)
    parser.add_argument("--dual-init", type=float, default=0.50)
    parser.add_argument("--dual-lr", type=float, default=5.0)
    parser.add_argument("--dual-max", type=float, default=8.0)
    parser.add_argument("--dual-decay", type=float, default=0.98)

    parser.add_argument("--epsilon", type=float, default=2.0 / 255.0)
    parser.add_argument("--pgd-alpha", type=float, default=0.5 / 255.0)
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--epsilon-warmup-epochs", type=int, default=20)
    parser.add_argument("--attack-top1-weight", type=float, default=0.25)
    parser.add_argument("--attack-set2-weight", type=float, default=0.75)

    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--robust-val-size", type=int, default=1000)
    parser.add_argument("--robust-eval-every", type=int, default=2)
    parser.add_argument("--robust-eval-steps", type=int, default=20)
    parser.add_argument("--robust-eval-alpha", type=float, default=0.5 / 255.0)
    parser.add_argument("--test-robust-size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=protocol.DEFAULT_SEED)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a checkpoint path, or use --resume for last_checkpoint.pt",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Two-epoch small pipeline check; not a scientific result",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_torchvision_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "cifar-100-python" and (path / "train").is_file():
        return path.parent
    nested = path / "cifar-100-python"
    if (nested / "train").is_file() and (nested / "test").is_file():
        return path
    raise FileNotFoundError(
        f"CIFAR-100 batches not found under '{path}' or '{nested}'."
    )


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
    def __init__(self, num_classes: int = 100):
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


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR100_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class DualBackboneDefense(nn.Module):
    def __init__(
        self,
        selector_temperature: float = 1.0,
        primary_margin: float = 1.0,
    ):
        super().__init__()
        self.robust_branch = CIFARResNet20(num_classes=100)
        self.set2_branch = CIFARResNet20(num_classes=100)
        self.selector_temperature = float(selector_temperature)
        self.primary_margin = float(primary_margin)
        self.register_buffer("mean", torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR100_STD).view(1, 3, 1, 1))

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
            maximum_base - primary_base + self.primary_margin
        ).clamp_min(self.primary_margin).detach()
        return set2_logits + straight_through * boost[:, None]

    def forward_all(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.normalize(x)
        robust_logits = self.robust_branch(normalized)
        set2_logits = self.set2_branch(normalized)
        final_logits = self.compose_logits(robust_logits, set2_logits)
        return robust_logits, set2_logits, final_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)[2]


def torch_load_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(loaded: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(loaded, Mapping):
        raise TypeError("Checkpoint is not a state_dict-like mapping")
    for key in ("model_state_dict", "state_dict", "model", "net"):
        candidate = loaded.get(key)
        if isinstance(candidate, Mapping) and candidate:
            loaded = candidate
            break
    state = dict(loaded)
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def checkpoint_body_state(
    path: Path, device: torch.device, description: str
) -> Dict[str, torch.Tensor]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    state = extract_state_dict(torch_load_compat(path, device))
    if "model.conv1.weight" in state:
        state = {
            key[len("model."):]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
    return {
        (
            "fc." + key[len("linear."):]
            if key.startswith("linear.")
            else key.replace(".shortcut.", ".downsample.")
        ): value
        for key, value in state.items()
        if not key.startswith("set2_head.")
    }


def load_body(
    model: CIFARResNet20,
    path: Path,
    device: torch.device,
    description: str,
) -> None:
    state = checkpoint_body_state(path, device, description)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{description}/model mismatch. Missing={list(missing)}; "
            f"unexpected={list(unexpected)}"
        )


def configure_trainable_layers(
    defense: DualBackboneDefense, selection: str
) -> Dict[str, torch.Tensor]:
    for parameter in defense.parameters():
        parameter.requires_grad_(False)
    if selection == "fc":
        modules = [defense.set2_branch.fc]
    elif selection == "layer3_fc":
        modules = [defense.set2_branch.layer3, defense.set2_branch.fc]
    elif selection == "all":
        modules = [defense.set2_branch]
    else:
        raise ValueError(f"Unknown trainable layer selection: {selection}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return {
        name: parameter.detach().clone()
        for name, parameter in defense.set2_branch.named_parameters()
        if parameter.requires_grad
    }


def stratified_train_val_indices(
    targets: Sequence[int], val_size: int, seed: int
) -> Tuple[list, list]:
    if val_size <= 0 or val_size >= len(targets) or val_size % 100 != 0:
        raise ValueError("--val-size must be divisible by 100 and smaller than 50000")
    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    per_class = val_size // 100
    train_indices, val_indices = [], []
    for class_id in range(100):
        indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(indices)
        val_indices.extend(indices[:per_class].tolist())
        train_indices.extend(indices[per_class:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def balanced_subset_from_indices(
    all_targets: Sequence[int], candidate_indices: Sequence[int], size: int
) -> list:
    if size <= 0 or size % 100 != 0:
        raise ValueError("Balanced subset size must be positive and divisible by 100")
    per_class = size // 100
    selected, counts = [], [0] * 100
    for index in candidate_indices:
        label = int(all_targets[index])
        if counts[label] < per_class:
            selected.append(int(index))
            counts[label] += 1
    if any(count != per_class for count in counts):
        raise ValueError(f"Could not form balanced subset; counts={counts}")
    return selected


def make_loaders(args: argparse.Namespace, device: torch.device):
    root = protocol.resolve_torchvision_root(args.data_dir)
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR100(
        root=str(root), train=True, download=False, transform=transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=str(root), train=False, download=False, transform=transform
    )
    train_indices, val_indices, split_payload = protocol.load_frozen_split(
        args.split_file, train_set.targets, args.seed, args.val_size
    )
    robust_val_indices = balanced_subset_from_indices(
        train_set.targets, val_indices, args.robust_val_size
    )
    robust_test_indices = balanced_subset_from_indices(
        test_set.targets, list(range(len(test_set))), args.test_robust_size
    )
    if args.quick:
        train_indices = balanced_subset_from_indices(train_set.targets, train_indices, 2000)
        val_indices = balanced_subset_from_indices(train_set.targets, val_indices, 500)
        robust_val_indices = balanced_subset_from_indices(train_set.targets, val_indices, 200)
        robust_test_indices = balanced_subset_from_indices(
            test_set.targets, list(range(len(test_set))), 200
        )
    common = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_set, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        Subset(train_set, val_indices),
        batch_size=max(256, args.batch_size),
        shuffle=False,
        **common,
    )
    robust_val_loader = DataLoader(
        Subset(train_set, robust_val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=max(256, args.batch_size),
        shuffle=False,
        **common,
    )
    robust_test_loader = DataLoader(
        Subset(test_set, robust_test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )
    return (
        train_loader,
        val_loader,
        robust_val_loader,
        test_loader,
        robust_test_loader,
        split_payload,
    )


def epsilon_for_epoch(epoch: int, target: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 1:
        return target
    return target * min(max(epoch / float(warmup_epochs), 0.0), 1.0)


def unordered_top2(logits: torch.Tensor) -> torch.Tensor:
    return logits.topk(2, dim=1).indices.sort(dim=1).values


def teacher_multihot(
    teacher_top2: torch.Tensor, num_classes: int = 100
) -> torch.Tensor:
    target = torch.zeros(
        teacher_top2.shape[0],
        num_classes,
        device=teacher_top2.device,
        dtype=torch.float32,
    )
    return target.scatter_(1, teacher_top2, 1.0)


def hard_set2_ranking_per_sample(
    logits: torch.Tensor,
    teacher_top2: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    selected = logits.gather(1, teacher_top2)
    weakest_selected = selected.min(dim=1).values
    outside_mask = torch.ones_like(logits, dtype=torch.bool)
    outside_mask.scatter_(1, teacher_top2, False)
    strongest_outside = logits.masked_fill(~outside_mask, float("-inf")).max(1).values
    return F.softplus(strongest_outside - weakest_selected + margin)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def branch_objective(
    robust_logits: torch.Tensor,
    set2_logits: torch.Tensor,
    final_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_top2: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    target = teacher_multihot(teacher_top2, set2_logits.shape[1]).to(set2_logits.dtype)
    positive_weight = torch.full(
        (set2_logits.shape[1],),
        args.positive_class_weight,
        device=set2_logits.device,
        dtype=set2_logits.dtype,
    )
    bce = F.binary_cross_entropy_with_logits(
        set2_logits, target, pos_weight=positive_weight
    )
    raw_ranking = hard_set2_ranking_per_sample(
        set2_logits, teacher_top2, args.set2_margin
    ).mean()
    primary = robust_logits.argmax(dim=1)
    eligible = teacher_top2.eq(primary[:, None]).any(1)
    composed_each = hard_set2_ranking_per_sample(
        final_logits, teacher_top2, args.set2_margin
    )
    if eligible.any():
        composed_ranking = composed_each[eligible].mean()
    else:
        composed_ranking = final_logits.sum() * 0.0
    kd = distillation_loss(set2_logits, teacher_logits, args.kd_temperature)
    total = (
        args.bce_weight * bce
        + args.raw_ranking_weight * raw_ranking
        + args.composed_ranking_weight * composed_ranking
        + args.kd_weight * kd
    )
    return total, {
        "bce": bce,
        "raw_ranking": raw_ranking,
        "composed_ranking": composed_ranking,
        "kd": kd,
        "eligible_rate": eligible.float().mean(),
    }


def weight_anchor_loss(
    defense: DualBackboneDefense,
    anchor: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, parameter in defense.set2_branch.named_parameters():
        if parameter.requires_grad:
            losses.append((parameter - anchor[name]).pow(2).mean())
    if not losses:
        raise RuntimeError("No trainable Set2 parameters for anchor loss")
    return torch.stack(losses).mean()


def pgd_attack(
    defense: DualBackboneDefense,
    images: torch.Tensor,
    labels: torch.Tensor,
    teacher_top2: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    args: argparse.Namespace,
    goal: str,
) -> torch.Tensor:
    if epsilon <= 0.0 or steps <= 0:
        return images.detach().clone()
    defense.eval()
    clean = images.detach()
    adversarial = torch.clamp(
        clean + torch.empty_like(clean).uniform_(-epsilon, epsilon), 0.0, 1.0
    )
    for _ in range(steps):
        adversarial.requires_grad_(True)
        with torch.enable_grad():
            robust_logits, _, final_logits = defense.forward_all(adversarial)
            top1_loss = F.cross_entropy(robust_logits, labels)
            set2_loss = hard_set2_ranking_per_sample(
                final_logits, teacher_top2, args.set2_margin
            ).mean()
            if goal == "top1":
                objective = top1_loss
            elif goal == "set2":
                objective = set2_loss
            elif goal == "joint":
                objective = (
                    args.attack_top1_weight * top1_loss
                    + args.attack_set2_weight * set2_loss
                )
            else:
                raise ValueError(f"Unknown attack goal: {goal}")
        gradient = torch.autograd.grad(objective, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0.0, 1.0).detach()
    return adversarial


@torch.inference_mode()
def evaluate_clean(
    defense: DualBackboneDefense,
    teacher: NormalizedModel,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:
    defense.eval()
    teacher.eval()
    top1_correct = 0
    final_set2_exact = 0
    branch_set2_exact = 0
    primary_in_teacher_set2 = 0
    label_in_final_set2 = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        teacher_top2 = unordered_top2(teacher(images))
        robust_logits, set2_logits, final_logits = defense.forward_all(images)
        primary = robust_logits.argmax(1)
        final_set2 = unordered_top2(final_logits)
        branch_set2 = unordered_top2(set2_logits)
        top1_correct += primary.eq(labels).sum().item()
        final_set2_exact += final_set2.eq(teacher_top2).all(1).sum().item()
        branch_set2_exact += branch_set2.eq(teacher_top2).all(1).sum().item()
        primary_in_teacher_set2 += teacher_top2.eq(primary[:, None]).any(1).sum().item()
        label_in_final_set2 += final_set2.eq(labels[:, None]).any(1).sum().item()
        total += labels.numel()
    return {
        "top1": 100.0 * top1_correct / total,
        "final_set2_exact": 100.0 * final_set2_exact / total,
        "set2_branch_exact": 100.0 * branch_set2_exact / total,
        "primary_in_teacher_set2_ceiling": 100.0 * primary_in_teacher_set2 / total,
        "label_in_final_set2": 100.0 * label_in_final_set2 / total,
    }


def evaluate_robust(
    defense: DualBackboneDefense,
    teacher: NormalizedModel,
    loader: Iterable,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    defense.eval()
    teacher.eval()
    top1_correct = 0
    final_set2_exact = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_top2 = unordered_top2(teacher(images))
        top1_adversarial = pgd_attack(
            defense, images, labels, teacher_top2,
            args.epsilon, args.robust_eval_alpha, args.robust_eval_steps,
            args, "top1"
        )
        set2_adversarial = pgd_attack(
            defense, images, labels, teacher_top2,
            args.epsilon, args.robust_eval_alpha, args.robust_eval_steps,
            args, "set2"
        )
        with torch.no_grad():
            attacked_top1 = defense.forward_all(top1_adversarial)[0].argmax(1)
            attacked_set2 = unordered_top2(defense.forward_all(set2_adversarial)[2])
            top1_correct += attacked_top1.eq(labels).sum().item()
            final_set2_exact += attacked_set2.eq(teacher_top2).all(1).sum().item()
        total += labels.numel()
    return {
        "robust_top1": 100.0 * top1_correct / total,
        "robust_set2_exact": 100.0 * final_set2_exact / total,
    }


def train_one_epoch(
    defense: DualBackboneDefense,
    teacher: NormalizedModel,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    current_epsilon: float,
    dual_value: float,
    anchor: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    # Eval mode deliberately freezes all BN running statistics while gradients
    # still update selected layer3/fc parameters.
    defense.eval()
    teacher.eval()
    loss_sum = 0.0
    clean_set2_correct = 0
    adversarial_set2_correct = 0
    adversarial_top1_correct = 0
    total = 0
    started = time.time()
    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_logits = teacher(images).float()
            teacher_top2 = unordered_top2(teacher_logits)
        adversarial = pgd_attack(
            defense,
            images,
            labels,
            teacher_top2,
            current_epsilon,
            min(args.pgd_alpha, max(current_epsilon / 4.0, 1e-12)),
            args.pgd_steps,
            args,
            "joint",
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            clean_outputs = defense.forward_all(images)
            adversarial_outputs = defense.forward_all(adversarial)
            clean_loss, _ = branch_objective(
                *clean_outputs, teacher_logits, teacher_top2, args
            )
            adversarial_loss, _ = branch_objective(
                *adversarial_outputs, teacher_logits, teacher_top2, args
            )
            anchor_loss = weight_anchor_loss(defense, anchor)
            loss = (
                args.clean_loss_weight * (1.0 + dual_value) * clean_loss
                + args.adversarial_loss_weight * adversarial_loss
                + args.anchor_weight * anchor_loss
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()
        total += batch_size
        loss_sum += loss.item() * batch_size
        clean_set2_correct += unordered_top2(clean_outputs[2]).eq(teacher_top2).all(1).sum().item()
        adversarial_set2_correct += unordered_top2(adversarial_outputs[2]).eq(teacher_top2).all(1).sum().item()
        adversarial_top1_correct += adversarial_outputs[0].argmax(1).eq(labels).sum().item()
        if batch_number == 1 or batch_number % 50 == 0 or batch_number == len(loader):
            print(
                f"epoch={epoch:03d} batch={batch_number:04d}/{len(loader):04d} "
                f"eps={current_epsilon:.5f} dual={dual_value:.3f} "
                f"loss={loss_sum/total:.4f} "
                f"clean_S2={100*clean_set2_correct/total:.2f}% "
                f"adv_S2={100*adversarial_set2_correct/total:.2f}% "
                f"adv_T1={100*adversarial_top1_correct/total:.2f}%",
                flush=True,
            )
    return {
        "train_loss": loss_sum / total,
        "train_clean_set2": 100.0 * clean_set2_correct / total,
        "train_adversarial_set2": 100.0 * adversarial_set2_correct / total,
        "train_adversarial_top1": 100.0 * adversarial_top1_correct / total,
        "epoch_seconds": time.time() - started,
    }


def update_dual(
    current: float,
    clean_set2: float,
    target: float,
    learning_rate: float,
    maximum: float,
    decay: float,
) -> float:
    violation = target - clean_set2
    if violation > 0.0:
        return min(maximum, current + learning_rate * violation / 100.0)
    return max(0.0, current * decay)


def feasible_score(clean_top1: float, robust_top1: float, robust_set2: float) -> float:
    return 0.20 * clean_top1 + 0.40 * robust_top1 + 0.40 * robust_set2


def composite_score(
    clean_top1: float,
    clean_set2: float,
    robust_top1: float,
    robust_set2: float,
) -> float:
    return 0.20 * clean_top1 + 0.20 * clean_set2 + 0.30 * robust_top1 + 0.30 * robust_set2


def append_metrics(path: Path, row: Dict[str, object]) -> None:
    # Upsert by epoch so a crash between CSV and checkpoint writes cannot
    # create duplicate epochs after --resume.
    previous = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            previous = [
                item
                for item in csv.DictReader(handle)
                if str(item.get("epoch")) != str(row["epoch"])
            ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(previous)
        writer.writerow(row)
    os.replace(temporary, path)


def atomic_torch_save(obj: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def save_raw_weights(model: nn.Module, path: Path) -> None:
    atomic_torch_save(model.state_dict(), path)


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train_loader: DataLoader,
) -> Tuple[int, Dict[str, object], float]:
    checkpoint = torch_load_compat(path, device)
    if checkpoint.get("protocol_id") != protocol.PROTOCOL_ID:
        raise ValueError("Resume checkpoint belongs to another protocol")
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "dual_value",
        "best",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint misses keys: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    rng_state = checkpoint.get("rng_state")
    if rng_state:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in rng_state["cuda"]]
            )
        train_loader.sampler.generator.set_state(rng_state["train_sampler"].cpu())
    return (
        int(checkpoint["epoch"]) + 1,
        dict(checkpoint["best"]),
        float(checkpoint["dual_value"]),
    )


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.clean_set2_constraint is not None
        and not 0.0 < args.clean_set2_constraint <= 100.0
    ):
        raise ValueError("--clean-set2-constraint must be in (0, 100] when provided")
    if not 0.0 <= args.clean_set2_max_drop <= 20.0:
        raise ValueError("--clean-set2-max-drop must be in [0, 20]")
    if args.lr <= 0 or args.selector_temperature <= 0 or args.primary_margin <= 0:
        raise ValueError("Learning rate, selector temperature and margin must be positive")
    weights = [
        args.anchor_weight, args.bce_weight, args.raw_ranking_weight,
        args.composed_ranking_weight, args.kd_weight, args.clean_loss_weight,
        args.adversarial_loss_weight, args.attack_top1_weight,
        args.attack_set2_weight,
    ]
    if any(weight < 0 for weight in weights):
        raise ValueError("All objective weights must be non-negative")
    if args.attack_top1_weight + args.attack_set2_weight <= 0:
        raise ValueError("At least one attack weight must be positive")
    if args.dual_init < 0 or args.dual_lr < 0 or args.dual_max <= 0:
        raise ValueError("Dual parameters must be non-negative and max positive")
    if not 0.0 <= args.dual_decay <= 1.0:
        raise ValueError("--dual-decay must be in [0, 1]")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 2
        args.epsilon_warmup_epochs = 2
        args.pgd_steps = min(args.pgd_steps, 2)
        args.robust_eval_steps = min(args.robust_eval_steps, 3)
        args.robust_eval_every = 1
        args.save_every = 0
    validate_args(args)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but no GPU is available")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        train_loader,
        val_loader,
        robust_val_loader,
        test_loader,
        robust_test_loader,
        split_payload,
    ) = make_loaders(args, device)
    teacher = NormalizedModel(CIFARResNet20()).to(device)
    defense = DualBackboneDefense(
        selector_temperature=args.selector_temperature,
        primary_margin=args.primary_margin,
    ).to(device)
    load_body(teacher.model, args.teacher_weights, device, "Teacher checkpoint")
    load_body(defense.robust_branch, args.robust_anchor_weights, device, "Robust anchor checkpoint")
    load_body(defense.set2_branch, args.teacher_weights, device, "Set2 initialization checkpoint")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    anchor = configure_trainable_layers(defense, args.trainable_set2_layers)
    trainable_parameters = [parameter for parameter in defense.parameters() if parameter.requires_grad]
    print("Frozen robust branch:", args.robust_anchor_weights)
    print("Set2 branch initialization:", args.teacher_weights)
    print("Trainable Set2 layers:", args.trainable_set2_layers)
    print("Trainable parameters:", f"{sum(p.numel() for p in trainable_parameters):,}")

    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    metrics_path = output_dir / "training_metrics.csv"

    initial_clean = evaluate_clean(defense, teacher, val_loader, device)
    if args.clean_set2_constraint is None:
        args.clean_set2_constraint = max(
            0.01,
            initial_clean["final_set2_exact"] - args.clean_set2_max_drop,
        )
        constraint_source = (
            "initial clean-validation Exact-Set2 minus "
            f"{args.clean_set2_max_drop:.3f} pp"
        )
    else:
        constraint_source = "explicit command-line value"
    frozen_protocol = {
        "protocol_id": protocol.PROTOCOL_ID,
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "selection_split": "held-out stratified training validation only",
        "test_used_for_selection": False,
        "initial_clean_exact_set": initial_clean["final_set2_exact"],
        "clean_set2_constraint": args.clean_set2_constraint,
        "constraint_source": constraint_source,
        "split_file": str(args.split_file.expanduser().resolve()),
        "train_indices_sha256": split_payload["train_indices_sha256"],
        "validation_indices_sha256": split_payload["validation_indices_sha256"],
        "selected_checkpoint_rule": (
            "maximum fixed validation composite among candidates satisfying "
            "the frozen clean Exact-Set2 constraint: 0.20 clean Top-1 + "
            "0.20 clean Exact-Set2 + 0.30 PGD Top-1 + 0.30 PGD Exact-Set2"
        ),
    }
    protocol.atomic_json(output_dir / "frozen_protocol.json", frozen_protocol)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["protocol_id"] = protocol.PROTOCOL_ID
    protocol.atomic_json(output_dir / "config.json", config)

    initial_robust = evaluate_robust(defense, teacher, robust_val_loader, device, args)
    initial_composite = composite_score(
        initial_clean["top1"], initial_clean["final_set2_exact"],
        initial_robust["robust_top1"], initial_robust["robust_set2_exact"]
    )
    initial_feasible = initial_clean["final_set2_exact"] >= args.clean_set2_constraint
    initial_feasible_score = feasible_score(
        initial_clean["top1"], initial_robust["robust_top1"],
        initial_robust["robust_set2_exact"]
    )
    last_checkpoint = output_dir / "last_checkpoint.pt"
    if args.resume is None:
        with (output_dir / "initial_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "clean": initial_clean,
                    "robust": initial_robust,
                    "composite": initial_composite,
                    "feasible": initial_feasible,
                    "feasible_score": initial_feasible_score if initial_feasible else None,
                },
                handle,
                indent=2,
            )
        save_raw_weights(defense, output_dir / "resnet20_dual_backbone_initial_weights.pt")
        save_raw_weights(defense, output_dir / "resnet20_dual_backbone_best_composite_weights.pt")
        best = {
            "clean_set2": initial_clean["final_set2_exact"],
            "robust_set2": initial_robust["robust_set2_exact"],
            "composite": initial_composite,
            "feasible_score": -1.0,
            "composite_epoch": 0,
            "composite_record": {
                "epoch": 0,
                "training_epsilon": 0.0,
                "validation_clean_top1": initial_clean["top1"],
                "validation_clean_set2": initial_clean["final_set2_exact"],
                "validation_robust_top1": initial_robust["robust_top1"],
                "validation_robust_set2": initial_robust["robust_set2_exact"],
                "validation_composite": initial_composite,
                "clean_set2_constraint": args.clean_set2_constraint,
                "constraint_satisfied": True,
                "selection_eligible": True,
                "note": "untrained composition, evaluated at full attack radius",
            },
        }
        if initial_feasible:
            best["feasible_score"] = initial_feasible_score
            save_raw_weights(
                defense,
                output_dir / "resnet20_dual_backbone_best_feasible_weights.pt",
            )
        start_epoch = 1
        dual_value = args.dual_init
    else:
        resume_path = (
            last_checkpoint
            if args.resume == "auto"
            else Path(args.resume).expanduser().resolve()
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best, dual_value = load_resume_checkpoint(
            resume_path,
            defense,
            optimizer,
            scheduler,
            scaler,
            device,
            train_loader,
        )
        print(f"Resumed from {resume_path}; next epoch={start_epoch}")
    print(
        "Initial dual-backbone | "
        f"clean_T1={initial_clean['top1']:.2f}% "
        f"clean_S2={initial_clean['final_set2_exact']:.2f}% "
        f"branch_S2={initial_clean['set2_branch_exact']:.2f}% "
        f"PGD_T1={initial_robust['robust_top1']:.2f}% "
        f"PGD_S2={initial_robust['robust_set2_exact']:.2f}% "
        f"feasible={initial_feasible}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        current_epsilon = epsilon_for_epoch(
            epoch, args.epsilon, args.epsilon_warmup_epochs
        )
        current_lr = optimizer.param_groups[0]["lr"]
        dual_before = dual_value
        train_metrics = train_one_epoch(
            defense, teacher, train_loader, optimizer, scaler, device, args,
            epoch, current_epsilon, dual_before, anchor
        )
        clean_metrics = evaluate_clean(defense, teacher, val_loader, device)
        dual_value = update_dual(
            dual_before, clean_metrics["final_set2_exact"],
            args.clean_set2_constraint, args.dual_lr, args.dual_max,
            args.dual_decay
        )
        run_robust_eval = epoch == 1 or epoch % args.robust_eval_every == 0 or epoch == args.epochs
        robust_metrics: Optional[Dict[str, float]] = None
        composite: Optional[float] = None
        current_feasible_score: Optional[float] = None
        if run_robust_eval:
            robust_metrics = evaluate_robust(
                defense, teacher, robust_val_loader, device, args
            )
            composite = composite_score(
                clean_metrics["top1"], clean_metrics["final_set2_exact"],
                robust_metrics["robust_top1"], robust_metrics["robust_set2_exact"]
            )
            if clean_metrics["final_set2_exact"] >= args.clean_set2_constraint:
                current_feasible_score = feasible_score(
                    clean_metrics["top1"], robust_metrics["robust_top1"],
                    robust_metrics["robust_set2_exact"]
                )
        selection_eligible = bool(
            composite is not None
            and clean_metrics["final_set2_exact"] >= args.clean_set2_constraint
        )
        scheduler.step()
        row: Dict[str, object] = {
            "epoch": epoch,
            "lr": current_lr,
            "training_epsilon": current_epsilon,
            "dual_before": dual_before,
            "dual_after": dual_value,
            **train_metrics,
            "validation_top1": clean_metrics["top1"],
            "validation_final_set2": clean_metrics["final_set2_exact"],
            "validation_branch_set2": clean_metrics["set2_branch_exact"],
            "validation_set2_ceiling": clean_metrics["primary_in_teacher_set2_ceiling"],
            "validation_robust_top1": "" if robust_metrics is None else robust_metrics["robust_top1"],
            "validation_robust_set2": "" if robust_metrics is None else robust_metrics["robust_set2_exact"],
            "validation_composite": "" if composite is None else composite,
            "validation_feasible_score": "" if current_feasible_score is None else current_feasible_score,
            "selection_eligible": selection_eligible,
        }
        append_metrics(metrics_path, row)

        if clean_metrics["final_set2_exact"] > best["clean_set2"]:
            best["clean_set2"] = clean_metrics["final_set2_exact"]
            save_raw_weights(defense, output_dir / "resnet20_dual_backbone_best_clean_set2_weights.pt")
        if robust_metrics is not None and robust_metrics["robust_set2_exact"] > best["robust_set2"]:
            best["robust_set2"] = robust_metrics["robust_set2_exact"]
            save_raw_weights(defense, output_dir / "resnet20_dual_backbone_best_robust_set2_weights.pt")
        if selection_eligible and composite > best["composite"]:
            best["composite"] = composite
            best["composite_epoch"] = epoch
            best["composite_record"] = {
                "epoch": epoch,
                "training_epsilon": current_epsilon,
                "validation_clean_top1": clean_metrics["top1"],
                "validation_clean_set2": clean_metrics["final_set2_exact"],
                "validation_robust_top1": robust_metrics["robust_top1"],
                "validation_robust_set2": robust_metrics["robust_set2_exact"],
                "validation_composite": composite,
                "clean_set2_constraint": args.clean_set2_constraint,
                "constraint_satisfied": True,
                "selection_eligible": True,
                "note": "full-radius validation attack used for selection",
            }
            save_raw_weights(defense, output_dir / "resnet20_dual_backbone_best_composite_weights.pt")
        if current_feasible_score is not None and current_feasible_score > best["feasible_score"]:
            best["feasible_score"] = current_feasible_score
            save_raw_weights(defense, output_dir / "resnet20_dual_backbone_best_feasible_weights.pt")

        checkpoint = {
            "protocol_id": protocol.PROTOCOL_ID,
            "epoch": epoch,
            "model_state_dict": defense.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "dual_value": dual_value,
            "best": best,
            "metrics": row,
            "args": vars(args),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "train_sampler": train_loader.sampler.generator.get_state(),
            },
        }
        atomic_torch_save(checkpoint, output_dir / "last_checkpoint.pt")
        save_raw_weights(defense, output_dir / "resnet20_dual_backbone_last_weights.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            atomic_torch_save(
                checkpoint, output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            )
        robust_text = "n/a" if robust_metrics is None else (
            f"T1={robust_metrics['robust_top1']:.2f}% "
            f"S2={robust_metrics['robust_set2_exact']:.2f}%"
        )
        print(
            f"epoch={epoch:03d} done | clean_T1={clean_metrics['top1']:.2f}% "
            f"clean_S2={clean_metrics['final_set2_exact']:.2f}% "
            f"branch_S2={clean_metrics['set2_branch_exact']:.2f}% | "
            f"robust {robust_text} | dual={dual_before:.3f}->{dual_value:.3f} "
            f"feasible={'yes' if current_feasible_score is not None else 'no'} "
            f"best_feasible={best['feasible_score']:.2f}",
            flush=True,
        )

    feasible_weights = output_dir / "resnet20_dual_backbone_best_feasible_weights.pt"
    composite_weights = output_dir / "resnet20_dual_backbone_best_composite_weights.pt"
    selected_weights = composite_weights
    if not selected_weights.is_file():
        raise RuntimeError("No selectable checkpoint was saved")
    selected_record = best.get("composite_record")
    if not isinstance(selected_record, dict):
        raise RuntimeError("Selected Dual checkpoint has no coherent validation record")
    selection_reason = (
        "maximum fixed validation composite among candidates satisfying the "
        "frozen clean Exact-Set2 floor: 0.20 clean Top-1 + 0.20 clean "
        "Exact-Set2 + 0.30 PGD Top-1 + 0.30 PGD Exact-Set2"
    )
    defense.load_state_dict(torch_load_compat(selected_weights, device))
    final_clean = evaluate_clean(defense, teacher, test_loader, device)
    final_robust = evaluate_robust(defense, teacher, robust_test_loader, device, args)
    final_metrics = {
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "PASS",
        "stage": "dual_top2",
        "selected_checkpoint": str(selected_weights),
        "selected_epoch": int(selected_record["epoch"]),
        "selected_validation": selected_record,
        "selection_reason": selection_reason,
        "test_clean": final_clean,
        "test_robust": final_robust,
        "robust_test_examples": 200 if args.quick else args.test_robust_size,
        "epsilon": args.epsilon,
        "pgd_steps": args.robust_eval_steps,
        "componentwise_validation_maxima_diagnostic": {
            key: value for key, value in best.items() if key != "composite_record"
        },
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "selection_split": "held-out stratified training validation only",
        "test_used_for_selection": False,
        "offline": True,
        "official_full_test": not args.quick,
        "split_file": str(args.split_file.expanduser().resolve()),
        "train_indices_sha256": split_payload["train_indices_sha256"],
        "validation_indices_sha256": split_payload["validation_indices_sha256"],
        "deployment": "frozen robust branch + trained Set2 branch; no reference teacher",
        "attack_note": "Top1 attacks robust logits; Set2 attacks straight-through composed logits",
    }
    with (output_dir / "final_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    print("Training complete.")
    print("Selected:", selected_weights)
    print("Reason:", selection_reason)
    print(f"Test clean Top-1:        {final_clean['top1']:.2f}%")
    print(f"Test clean final Set2:   {final_clean['final_set2_exact']:.2f}%")
    print(f"Test Set2 branch exact:  {final_clean['set2_branch_exact']:.2f}%")
    print(f"Clean Set2 ceiling:      {final_clean['primary_in_teacher_set2_ceiling']:.2f}%")
    print(f"Test PGD Top-1:          {final_robust['robust_top1']:.2f}%")
    print(f"Test PGD final Set2:     {final_robust['robust_set2_exact']:.2f}%")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
