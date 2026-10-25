#!/usr/bin/env python3
"""Teacher-guided adversarial training for CIFAR-10 ResNet-20.

Research goal
-------------
The high-clean-accuracy standard model is used only during training as a
frozen teacher.  Its clean-image Top-2 set supervises a robust student that is
initialized from the mixed-PGD checkpoint.  At inference time only the student
is needed; therefore there is no attackable standard-model selector, gate, or
post-hoc logit bias.

The student is trained with six terms:

* clean and adversarial cross-entropy for true-label Top-1;
* clean and adversarial knowledge distillation from teacher logits;
* clean and adversarial Top-2 set-margin losses.

The adversarial examples themselves jointly attack true-label Top-1 and the
teacher Top-2 set.  Checkpoint selection uses a held-out training validation
split.  In particular, ``best_feasible`` is the most robust checkpoint whose
clean validation Set2 reaches ``--clean-set2-target`` (80% by default).

All perturbation budgets are in raw RGB pixel space [0, 1].  CIFAR-10
normalization is inside the model.
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


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher-Top2 adversarial training of CIFAR-10 ResNet-20"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "./datasets/cifar-10-batches-py"
        ),
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=Path(
            "./models/resnet20/cifar10/"
            "resnet20_cifar10.pth"
        ),
        help="Standard ResNet-20 (~92.9%% clean accuracy); frozen teacher",
    )
    parser.add_argument(
        "--student-init-weights",
        type=Path,
        default=Path(
            "./models/resnet20/mixed_pgd_at/"
            "resnet20_mixed_pgd_best_composite_weights.pt"
        ),
        help="Mixed-PGD checkpoint used to initialize the student",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./models/resnet20/teacher_top2_at"
        ),
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--set2-margin", type=float, default=0.50)

    # Defaults sum to 1.0 and deliberately prioritize adversarial Set2.
    parser.add_argument("--clean-ce-weight", type=float, default=0.20)
    parser.add_argument("--adversarial-ce-weight", type=float, default=0.30)
    parser.add_argument("--clean-kd-weight", type=float, default=0.05)
    parser.add_argument("--adversarial-kd-weight", type=float, default=0.05)
    parser.add_argument("--clean-set2-weight", type=float, default=0.15)
    parser.add_argument("--adversarial-set2-weight", type=float, default=0.25)

    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--pgd-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--epsilon-warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--attack-top1-weight",
        type=float,
        default=0.50,
        help="Top-1 term in the training PGD objective",
    )
    parser.add_argument(
        "--attack-set2-weight",
        type=float,
        default=0.50,
        help="teacher-Set2 violation term in the training PGD objective",
    )

    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--robust-val-size", type=int, default=1000)
    parser.add_argument("--robust-eval-every", type=int, default=5)
    parser.add_argument("--robust-eval-steps", type=int, default=20)
    parser.add_argument("--robust-eval-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--test-robust-size", type=int, default=1000)
    parser.add_argument(
        "--clean-set2-target",
        type=float,
        default=80.0,
        help="Minimum clean validation Set2 for the best-feasible checkpoint",
    )
    parser.add_argument("--selection-clean-top1-weight", type=float, default=0.20)
    parser.add_argument("--selection-clean-set2-weight", type=float, default=0.20)
    parser.add_argument("--selection-robust-top1-weight", type=float, default=0.30)
    parser.add_argument("--selection-robust-set2-weight", type=float, default=0.30)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
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
        help="Pipeline check: 2 epochs, small balanced splits, short PGD",
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
    if path.name == "cifar-10-batches-py" and (path / "data_batch_1").is_file():
        return path.parent
    nested = path / "cifar-10-batches-py"
    if (nested / "data_batch_1").is_file():
        return path
    raise FileNotFoundError(
        f"CIFAR-10 batches not found under '{path}' or '{nested}'."
    )


class ProjectionBasicBlock(nn.Module):
    """CIFAR block with learned projection shortcuts named ``downsample``."""

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


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def make_model() -> NormalizedModel:
    return NormalizedModel(CIFARResNet20(num_classes=10))


def torch_load_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(loaded: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(loaded, Mapping):
        raise TypeError("Checkpoint is not a state_dict-like mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = loaded.get(key)
        if isinstance(candidate, Mapping) and candidate:
            loaded = candidate
            break
    state = dict(loaded)
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def load_model_weights(
    model: NormalizedModel, path: Path, device: torch.device, description: str
) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    state = extract_state_dict(torch_load_compat(path, device))
    if "model.conv1.weight" in state:
        body_state = {
            key[len("model."):]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
    else:
        body_state = state
    body_state = {
        key.replace(".shortcut.", ".downsample."): value
        for key, value in body_state.items()
    }
    missing, unexpected = model.model.load_state_dict(body_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{description}/model mismatch. Missing={list(missing)}; "
            f"unexpected={list(unexpected)}"
        )


def stratified_train_val_indices(
    targets: Sequence[int], val_size: int, seed: int
) -> Tuple[list, list]:
    if val_size <= 0 or val_size >= len(targets) or val_size % 10 != 0:
        raise ValueError("--val-size must be divisible by 10 and smaller than 50000")
    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    per_class = val_size // 10
    train_indices, val_indices = [], []
    for class_id in range(10):
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
    if size <= 0 or size % 10 != 0:
        raise ValueError("Balanced subset size must be positive and divisible by 10")
    per_class = size // 10
    selected, counts = [], [0] * 10
    for index in candidate_indices:
        label = int(all_targets[index])
        if counts[label] < per_class:
            selected.append(int(index))
            counts[label] += 1
    if any(count != per_class for count in counts):
        raise ValueError(f"Could not form balanced subset; counts={counts}")
    return selected


def make_loaders(args: argparse.Namespace, device: torch.device):
    root = resolve_torchvision_root(args.data_dir)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    plain_transform = transforms.ToTensor()
    train_aug = torchvision.datasets.CIFAR10(
        root=str(root), train=True, download=False, transform=train_transform
    )
    train_plain = torchvision.datasets.CIFAR10(
        root=str(root), train=True, download=False, transform=plain_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(root), train=False, download=False, transform=plain_transform
    )
    train_indices, val_indices = stratified_train_val_indices(
        train_aug.targets, args.val_size, args.seed
    )
    robust_val_indices = balanced_subset_from_indices(
        train_aug.targets, val_indices, args.robust_val_size
    )
    robust_test_indices = balanced_subset_from_indices(
        test_set.targets, list(range(len(test_set))), args.test_robust_size
    )
    if args.quick:
        train_indices = balanced_subset_from_indices(
            train_aug.targets, train_indices, 2000
        )
        val_indices = balanced_subset_from_indices(train_aug.targets, val_indices, 500)
        robust_val_indices = balanced_subset_from_indices(
            train_aug.targets, val_indices, 200
        )
        robust_test_indices = balanced_subset_from_indices(
            test_set.targets, list(range(len(test_set))), 200
        )

    generator = torch.Generator().manual_seed(args.seed)
    common = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        Subset(train_aug, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        Subset(train_plain, val_indices),
        batch_size=max(256, args.batch_size),
        shuffle=False,
        **common,
    )
    robust_val_loader = DataLoader(
        Subset(train_plain, robust_val_indices),
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
    return train_loader, val_loader, robust_val_loader, test_loader, robust_test_loader


def epsilon_for_epoch(epoch: int, target: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 1:
        return target
    progress = min(max((epoch - 1) / float(warmup_epochs - 1), 0.0), 1.0)
    return target * progress


def unordered_top2(logits: torch.Tensor) -> torch.Tensor:
    return logits.topk(2, dim=1).indices.sort(dim=1).values


def set2_margin_loss(
    logits: torch.Tensor, teacher_top2: torch.Tensor, margin: float
) -> torch.Tensor:
    """Require both teacher-selected logits to outrank every other class."""
    selected = logits.gather(1, teacher_top2)
    weakest_selected = selected.min(dim=1).values
    outside_mask = torch.ones_like(logits, dtype=torch.bool)
    outside_mask.scatter_(1, teacher_top2, False)
    strongest_outside = logits.masked_fill(~outside_mask, float("-inf")).max(1).values
    return F.softplus(strongest_outside - weakest_selected + margin).mean()


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


def joint_pgd_attack(
    student: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    teacher_top2: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    top1_weight: float,
    set2_weight: float,
    set2_margin: float,
    goal: str = "joint",
) -> torch.Tensor:
    """White-box PGD maximizing Top-1 loss, Set2 loss, or both."""
    if epsilon <= 0.0 or steps <= 0:
        return images.detach().clone()
    was_training = student.training
    student.eval()
    clean = images.detach()
    adversarial = torch.clamp(
        clean + torch.empty_like(clean).uniform_(-epsilon, epsilon), 0.0, 1.0
    )
    for _ in range(steps):
        adversarial.requires_grad_(True)
        with torch.enable_grad():
            logits = student(adversarial)
            top1_term = F.cross_entropy(logits, labels)
            set2_term = set2_margin_loss(logits, teacher_top2, set2_margin)
            if goal == "top1":
                objective = top1_term
            elif goal == "set2":
                objective = set2_term
            elif goal == "joint":
                objective = top1_weight * top1_term + set2_weight * set2_term
            else:
                raise ValueError(f"Unknown attack goal: {goal}")
        gradient = torch.autograd.grad(objective, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0.0, 1.0).detach()
    student.train(was_training)
    return adversarial


@torch.inference_mode()
def evaluate_clean(
    student: nn.Module,
    teacher: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:
    student.eval()
    teacher.eval()
    student_correct = 0
    teacher_correct = 0
    set2_exact = 0
    label_in_student_set2 = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        student_logits = student(images)
        teacher_logits = teacher(images)
        student_set2 = unordered_top2(student_logits)
        teacher_set2 = unordered_top2(teacher_logits)
        student_correct += student_logits.argmax(1).eq(labels).sum().item()
        teacher_correct += teacher_logits.argmax(1).eq(labels).sum().item()
        set2_exact += student_set2.eq(teacher_set2).all(1).sum().item()
        label_in_student_set2 += student_set2.eq(labels[:, None]).any(1).sum().item()
        total += labels.numel()
    return {
        "student_top1": 100.0 * student_correct / total,
        "teacher_top1": 100.0 * teacher_correct / total,
        "set2_exact": 100.0 * set2_exact / total,
        "label_in_student_set2": 100.0 * label_in_student_set2 / total,
    }


def evaluate_robust(
    student: nn.Module,
    teacher: nn.Module,
    loader: Iterable,
    device: torch.device,
    epsilon: float,
    alpha: float,
    steps: int,
    set2_margin: float,
) -> Dict[str, float]:
    """Use separate strong white-box attacks for Top-1 and exact Set2."""
    student.eval()
    teacher.eval()
    top1_correct = 0
    set2_exact = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_top2 = unordered_top2(teacher(images))
        top1_adversarial = joint_pgd_attack(
            student,
            images,
            labels,
            teacher_top2,
            epsilon,
            alpha,
            steps,
            top1_weight=1.0,
            set2_weight=0.0,
            set2_margin=set2_margin,
            goal="top1",
        )
        set2_adversarial = joint_pgd_attack(
            student,
            images,
            labels,
            teacher_top2,
            epsilon,
            alpha,
            steps,
            top1_weight=0.0,
            set2_weight=1.0,
            set2_margin=set2_margin,
            goal="set2",
        )
        with torch.no_grad():
            top1_correct += student(top1_adversarial).argmax(1).eq(labels).sum().item()
            attacked_set2 = unordered_top2(student(set2_adversarial))
            set2_exact += attacked_set2.eq(teacher_top2).all(1).sum().item()
        total += labels.numel()
    return {
        "robust_top1": 100.0 * top1_correct / total,
        "robust_set2_exact": 100.0 * set2_exact / total,
    }


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    current_epsilon: float,
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    loss_sum = 0.0
    clean_top1_correct = 0
    adversarial_top1_correct = 0
    clean_set2_exact = 0
    adversarial_set2_exact = 0
    total = 0
    started = time.time()
    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_logits = teacher(images).float()
            teacher_top2 = unordered_top2(teacher_logits)

        adversarial = joint_pgd_attack(
            student,
            images,
            labels,
            teacher_top2,
            epsilon=current_epsilon,
            alpha=min(args.pgd_alpha, max(current_epsilon, 1e-12)),
            steps=args.pgd_steps,
            top1_weight=args.attack_top1_weight,
            set2_weight=args.attack_set2_weight,
            set2_margin=args.set2_margin,
            goal="joint",
        )
        student.train()
        optimizer.zero_grad(set_to_none=True)
        mixed_images = torch.cat([images, adversarial], dim=0)
        with torch.cuda.amp.autocast(enabled=args.amp):
            mixed_logits = student(mixed_images)
            clean_logits, adversarial_logits = mixed_logits.chunk(2, dim=0)
            clean_ce = F.cross_entropy(clean_logits, labels)
            adversarial_ce = F.cross_entropy(adversarial_logits, labels)
            clean_kd = distillation_loss(
                clean_logits, teacher_logits, args.kd_temperature
            )
            adversarial_kd = distillation_loss(
                adversarial_logits, teacher_logits, args.kd_temperature
            )
            clean_set2 = set2_margin_loss(
                clean_logits, teacher_top2, args.set2_margin
            )
            adversarial_set2 = set2_margin_loss(
                adversarial_logits, teacher_top2, args.set2_margin
            )
            loss = (
                args.clean_ce_weight * clean_ce
                + args.adversarial_ce_weight * adversarial_ce
                + args.clean_kd_weight * clean_kd
                + args.adversarial_kd_weight * adversarial_kd
                + args.clean_set2_weight * clean_set2
                + args.adversarial_set2_weight * adversarial_set2
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()
        total += batch_size
        loss_sum += loss.item() * batch_size
        clean_top1_correct += clean_logits.argmax(1).eq(labels).sum().item()
        adversarial_top1_correct += adversarial_logits.argmax(1).eq(labels).sum().item()
        clean_set2_exact += unordered_top2(clean_logits).eq(teacher_top2).all(1).sum().item()
        adversarial_set2_exact += unordered_top2(adversarial_logits).eq(teacher_top2).all(1).sum().item()
        if batch_number == 1 or batch_number % 50 == 0 or batch_number == len(loader):
            print(
                f"epoch={epoch:03d} batch={batch_number:04d}/{len(loader):04d} "
                f"eps={current_epsilon:.5f} loss={loss_sum/total:.4f} "
                f"clean_T1={100*clean_top1_correct/total:.2f}% "
                f"clean_S2={100*clean_set2_exact/total:.2f}% "
                f"adv_T1={100*adversarial_top1_correct/total:.2f}% "
                f"adv_S2={100*adversarial_set2_exact/total:.2f}%",
                flush=True,
            )
    return {
        "train_loss": loss_sum / total,
        "train_clean_top1": 100.0 * clean_top1_correct / total,
        "train_clean_set2": 100.0 * clean_set2_exact / total,
        "train_adversarial_top1": 100.0 * adversarial_top1_correct / total,
        "train_adversarial_set2": 100.0 * adversarial_set2_exact / total,
        "epoch_seconds": time.time() - started,
    }


def atomic_torch_save(obj: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def save_raw_weights(model: nn.Module, path: Path) -> None:
    atomic_torch_save(model.state_dict(), path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best: Dict[str, float],
    metrics: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    atomic_torch_save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best": best,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, Dict[str, float]]:
    checkpoint = torch_load_compat(path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    default_best = {
        "clean_top1": -1.0,
        "clean_set2": -1.0,
        "robust_top1": -1.0,
        "robust_set2": -1.0,
        "composite": -1.0,
        "feasible_score": -1.0,
    }
    default_best.update(checkpoint.get("best", {}))
    return int(checkpoint["epoch"]) + 1, default_best


def append_metrics(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_args(args: argparse.Namespace) -> None:
    loss_weights = [
        args.clean_ce_weight,
        args.adversarial_ce_weight,
        args.clean_kd_weight,
        args.adversarial_kd_weight,
        args.clean_set2_weight,
        args.adversarial_set2_weight,
    ]
    if any(weight < 0 for weight in loss_weights) or sum(loss_weights) <= 0:
        raise ValueError("Training loss weights must be non-negative with positive sum")
    attack_weight_sum = args.attack_top1_weight + args.attack_set2_weight
    if args.attack_top1_weight < 0 or args.attack_set2_weight < 0 or attack_weight_sum <= 0:
        raise ValueError("Attack weights must be non-negative with positive sum")
    selection_weights = [
        args.selection_clean_top1_weight,
        args.selection_clean_set2_weight,
        args.selection_robust_top1_weight,
        args.selection_robust_set2_weight,
    ]
    if any(weight < 0 for weight in selection_weights) or sum(selection_weights) <= 0:
        raise ValueError("Selection weights must be non-negative with positive sum")
    if args.robust_eval_every < 1:
        raise ValueError("--robust-eval-every must be >= 1")
    if args.kd_temperature <= 0 or args.set2_margin < 0:
        raise ValueError("Temperature must be positive and margin non-negative")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 2
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
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            handle,
            indent=2,
        )

    train_loader, val_loader, robust_val_loader, test_loader, robust_test_loader = make_loaders(args, device)
    teacher = make_model().to(device)
    student = make_model().to(device)
    load_model_weights(teacher, args.teacher_weights, device, "Teacher checkpoint")
    load_model_weights(student, args.student_init_weights, device, "Student initialization")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print("Frozen teacher:", args.teacher_weights)
    print("Student initialization:", args.student_init_weights)
    print("Student parameters:", f"{sum(p.numel() for p in student.parameters()):,}")

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    last_checkpoint = output_dir / "last_checkpoint.pt"
    metrics_path = output_dir / "training_metrics.csv"
    best = {
        "clean_top1": -1.0,
        "clean_set2": -1.0,
        "robust_top1": -1.0,
        "robust_set2": -1.0,
        "composite": -1.0,
        "feasible_score": -1.0,
    }
    start_epoch = 1
    if args.resume is not None:
        resume_path = last_checkpoint if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best = load_resume_checkpoint(
            resume_path, student, optimizer, scheduler, scaler, device
        )
        print(f"Resumed from {resume_path}; next epoch={start_epoch}")
    else:
        initial_clean = evaluate_clean(student, teacher, val_loader, device)
        initial_robust = evaluate_robust(
            student,
            teacher,
            robust_val_loader,
            device,
            args.epsilon,
            args.robust_eval_alpha,
            args.robust_eval_steps,
            args.set2_margin,
        )
        initial = {**initial_clean, **initial_robust}
        with (output_dir / "initial_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(initial, handle, indent=2)
        print(
            "Initial student | "
            f"clean_T1={initial_clean['student_top1']:.2f}% "
            f"clean_S2={initial_clean['set2_exact']:.2f}% "
            f"PGD_T1={initial_robust['robust_top1']:.2f}% "
            f"PGD_S2={initial_robust['robust_set2_exact']:.2f}%"
        )
        print(f"Teacher clean validation Top-1: {initial_clean['teacher_top1']:.2f}%")

    selection_weight_sum = (
        args.selection_clean_top1_weight
        + args.selection_clean_set2_weight
        + args.selection_robust_top1_weight
        + args.selection_robust_set2_weight
    )
    for epoch in range(start_epoch, args.epochs + 1):
        current_epsilon = epsilon_for_epoch(epoch, args.epsilon, args.epsilon_warmup_epochs)
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            epoch,
            current_epsilon,
        )
        clean_metrics = evaluate_clean(student, teacher, val_loader, device)
        run_robust_eval = epoch == 1 or epoch % args.robust_eval_every == 0 or epoch == args.epochs
        robust_metrics: Optional[Dict[str, float]] = None
        composite: Optional[float] = None
        feasible_score: Optional[float] = None
        if run_robust_eval:
            robust_metrics = evaluate_robust(
                student,
                teacher,
                robust_val_loader,
                device,
                args.epsilon,
                args.robust_eval_alpha,
                args.robust_eval_steps,
                args.set2_margin,
            )
            composite = (
                args.selection_clean_top1_weight * clean_metrics["student_top1"]
                + args.selection_clean_set2_weight * clean_metrics["set2_exact"]
                + args.selection_robust_top1_weight * robust_metrics["robust_top1"]
                + args.selection_robust_set2_weight * robust_metrics["robust_set2_exact"]
            ) / selection_weight_sum
            if clean_metrics["set2_exact"] >= args.clean_set2_target:
                # Once the clean Set2 constraint is met, prioritize attacked metrics.
                feasible_score = (
                    0.20 * clean_metrics["student_top1"]
                    + 0.40 * robust_metrics["robust_top1"]
                    + 0.40 * robust_metrics["robust_set2_exact"]
                )
        scheduler.step()

        row: Dict[str, object] = {
            "epoch": epoch,
            "lr": current_lr,
            "training_epsilon": current_epsilon,
            **train_metrics,
            "validation_clean_top1": clean_metrics["student_top1"],
            "validation_teacher_top1": clean_metrics["teacher_top1"],
            "validation_clean_set2": clean_metrics["set2_exact"],
            "validation_label_in_student_set2": clean_metrics["label_in_student_set2"],
            "validation_robust_top1": "" if robust_metrics is None else robust_metrics["robust_top1"],
            "validation_robust_set2": "" if robust_metrics is None else robust_metrics["robust_set2_exact"],
            "validation_composite": "" if composite is None else composite,
            "validation_feasible_score": "" if feasible_score is None else feasible_score,
        }
        append_metrics(metrics_path, row)

        if clean_metrics["student_top1"] > best["clean_top1"]:
            best["clean_top1"] = clean_metrics["student_top1"]
            save_raw_weights(student, output_dir / "resnet20_teacher_top2_best_clean_top1_weights.pt")
        if clean_metrics["set2_exact"] > best["clean_set2"]:
            best["clean_set2"] = clean_metrics["set2_exact"]
            save_raw_weights(student, output_dir / "resnet20_teacher_top2_best_clean_set2_weights.pt")
        if robust_metrics is not None and robust_metrics["robust_top1"] > best["robust_top1"]:
            best["robust_top1"] = robust_metrics["robust_top1"]
            save_raw_weights(student, output_dir / "resnet20_teacher_top2_best_robust_top1_weights.pt")
        if robust_metrics is not None and robust_metrics["robust_set2_exact"] > best["robust_set2"]:
            best["robust_set2"] = robust_metrics["robust_set2_exact"]
            save_raw_weights(student, output_dir / "resnet20_teacher_top2_best_robust_set2_weights.pt")
        if composite is not None and composite > best["composite"]:
            best["composite"] = composite
            save_raw_weights(student, output_dir / "resnet20_teacher_top2_best_composite_weights.pt")
        if feasible_score is not None and feasible_score > best["feasible_score"]:
            best["feasible_score"] = feasible_score
            save_raw_weights(student, output_dir / "resnet20_teacher_top2_best_feasible_weights.pt")

        save_checkpoint(
            last_checkpoint,
            student,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best,
            row,
            args,
        )
        save_raw_weights(student, output_dir / "resnet20_teacher_top2_last_weights.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                student,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best,
                row,
                args,
            )

        robust_text = "n/a" if robust_metrics is None else (
            f"T1={robust_metrics['robust_top1']:.2f}% "
            f"S2={robust_metrics['robust_set2_exact']:.2f}%"
        )
        feasible_text = "no" if feasible_score is None else f"yes/{feasible_score:.2f}"
        print(
            f"epoch={epoch:03d} done | clean_T1={clean_metrics['student_top1']:.2f}% "
            f"clean_S2={clean_metrics['set2_exact']:.2f}% | robust {robust_text} | "
            f"feasible={feasible_text} best_composite={best['composite']:.2f}",
            flush=True,
        )

    feasible_weights = output_dir / "resnet20_teacher_top2_best_feasible_weights.pt"
    composite_weights = output_dir / "resnet20_teacher_top2_best_composite_weights.pt"
    selected_weights = feasible_weights if feasible_weights.is_file() else composite_weights
    selection_reason = (
        f"best validation robustness subject to clean Set2 >= {args.clean_set2_target:.2f}%"
        if feasible_weights.is_file()
        else "best composite; no checkpoint met the clean Set2 constraint"
    )
    if not selected_weights.is_file():
        raise RuntimeError("No selectable weights were saved")
    student.load_state_dict(torch_load_compat(selected_weights, device))
    final_clean = evaluate_clean(student, teacher, test_loader, device)
    final_robust = evaluate_robust(
        student,
        teacher,
        robust_test_loader,
        device,
        args.epsilon,
        args.robust_eval_alpha,
        args.robust_eval_steps,
        args.set2_margin,
    )
    final_metrics = {
        "selected_checkpoint": str(selected_weights),
        "selection_reason": selection_reason,
        "test_clean": final_clean,
        "test_robust": final_robust,
        "robust_test_examples": 200 if args.quick else args.test_robust_size,
        "epsilon": args.epsilon,
        "pgd_steps": args.robust_eval_steps,
        "best_validation": best,
        "set2_definition": "exact unordered match to frozen teacher clean-image Top-2",
    }
    with (output_dir / "final_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    print("Training complete.")
    print("Selected:", selected_weights)
    print("Reason:", selection_reason)
    print(f"Test clean Top-1: {final_clean['student_top1']:.2f}%")
    print(f"Test clean Set2:  {final_clean['set2_exact']:.2f}%")
    print(f"Test PGD Top-1:   {final_robust['robust_top1']:.2f}%")
    print(f"Test PGD Set2:    {final_robust['robust_set2_exact']:.2f}%")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
