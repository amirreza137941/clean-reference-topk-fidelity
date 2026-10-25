#!/usr/bin/env python3
"""Leakage-free standard CIFAR-100 ResNet-20 training for protocol R2.

Unlike R1, this reference is initialized randomly and trained on only the
frozen 45,000-example training partition.  Its checkpoint is selected on the
disjoint frozen 5,000-example validation partition.  The official test set is
evaluated exactly once after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import topk_cifar100_r2_common as protocol
import train_resnet20_teacher_top2_adversarial_cifar100_r2 as models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the leakage-free CIFAR-100 R2 standard reference"
    )
    parser.add_argument("--data-dir", type=Path, default=protocol.DEFAULT_DATA)
    parser.add_argument(
        "--split-file", type=Path,
        default=protocol.DEFAULT_ROOT / "frozen_split.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=protocol.DEFAULT_ROOT / "standard"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--val-size", type=int, default=protocol.DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=protocol.DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume a complete optimizer/RNG state (default: last_checkpoint.pt)",
    )
    parser.add_argument("--quick", action="store_true")
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


def balanced_subset(
    targets: list[int], candidate_indices: list[int], size: int
) -> list[int]:
    return protocol.balanced_subset_from_indices(targets, candidate_indices, size)


def make_loaders(args: argparse.Namespace, device: torch.device):
    root = protocol.resolve_torchvision_root(args.data_dir)
    augmented = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    plain = transforms.ToTensor()
    train_aug = torchvision.datasets.CIFAR100(
        root=str(root), train=True, download=False, transform=augmented
    )
    train_plain = torchvision.datasets.CIFAR100(
        root=str(root), train=True, download=False, transform=plain
    )
    test_set = torchvision.datasets.CIFAR100(
        root=str(root), train=False, download=False, transform=plain
    )
    train_indices, validation_indices, split = protocol.load_frozen_split(
        args.split_file, train_aug.targets, args.seed, args.val_size
    )
    test_indices: list[int] | None = None
    if args.quick:
        train_indices = balanced_subset(train_aug.targets, train_indices, 2000)
        validation_indices = balanced_subset(
            train_plain.targets, validation_indices, 500
        )
        test_indices = balanced_subset(
            test_set.targets, list(range(len(test_set))), 500
        )
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        Subset(train_aug, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=generator, **common,
    )
    validation_loader = DataLoader(
        Subset(train_plain, validation_indices), batch_size=args.eval_batch_size,
        shuffle=False, **common,
    )
    test_source = test_set if test_indices is None else Subset(test_set, test_indices)
    test_loader = DataLoader(
        test_source, batch_size=args.eval_batch_size, shuffle=False, **common
    )
    return train_loader, validation_loader, test_loader, split


def cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    # ``epoch`` is zero based; this matches a standard full-period cosine decay.
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))


def set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    label_smoothing: float,
) -> dict[str, float]:
    model.train()
    started = time.time()
    total = correct = 0
    loss_sum = 0.0
    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(images)
            loss = F.cross_entropy(
                logits, labels, label_smoothing=label_smoothing
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.numel()
        total += batch_size
        correct += logits.detach().argmax(1).eq(labels).sum().item()
        loss_sum += float(loss.detach().item()) * batch_size
        if batch_number == 1 or batch_number % 100 == 0 or batch_number == len(loader):
            print(
                f"standard epoch={epoch:03d} batch={batch_number:04d}/{len(loader):04d} "
                f"loss={loss_sum/total:.4f} top1={100.0*correct/total:.2f}%",
                flush=True,
            )
    return {
        "train_loss": loss_sum / total,
        "train_top1": 100.0 * correct / total,
        "epoch_seconds": time.time() - started,
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float | int]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    finite = True
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        finite = finite and bool(torch.isfinite(logits).all().item())
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total += labels.numel()
        correct += logits.argmax(1).eq(labels).sum().item()
        loss_sum += float(loss.item())
    if not finite or total == 0:
        raise AssertionError("Standard evaluation produced invalid logits/results")
    return {
        "examples": total,
        "loss": loss_sum / total,
        "top1": 100.0 * correct / total,
        "correct": correct,
    }


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    previous: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            previous = [
                item for item in csv.DictReader(handle)
                if str(item.get("epoch")) != str(row["epoch"])
            ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(previous)
        writer.writerow(row)
    os.replace(temporary, path)


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    train_loader: DataLoader,
    epoch: int,
    best_record: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    atomic_torch_save(
        {
            "protocol_id": protocol.PROTOCOL_ID,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_record": best_record,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "train_sampler": train_loader.sampler.generator.get_state(),
            },
        },
        path,
    )


def load_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    train_loader: DataLoader,
    device: torch.device,
) -> tuple[int, dict[str, Any] | None]:
    state = models.torch_load_compat(path, device)
    if state.get("protocol_id") != protocol.PROTOCOL_ID:
        raise ValueError("Resume checkpoint belongs to another protocol")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])
    rng = state.get("rng_state")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in rng["cuda"]])
        train_loader.sampler.generator.set_state(rng["train_sampler"].cpu())
    return int(state["epoch"]) + 1, state.get("best_record")


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    if float(candidate["validation_top1"]) > float(best["validation_top1"]) + 1e-12:
        return True
    return (
        abs(float(candidate["validation_top1"]) - float(best["validation_top1"])) <= 1e-12
        and float(candidate["validation_loss"]) < float(best["validation_loss"])
    )


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 2
        args.save_every = 0
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0,1)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    config["protocol_id"] = protocol.PROTOCOL_ID
    protocol.atomic_json(output_dir / "config.json", config)

    train_loader, validation_loader, test_loader, split = make_loaders(args, device)
    model = models.make_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    last_checkpoint = output_dir / "last_checkpoint.pt"
    selected_weights = output_dir / "resnet20_standard_best_validation_weights.pt"
    metrics_path = output_dir / "training_metrics.csv"
    start_epoch = 1
    best_record: dict[str, Any] | None = None
    if args.resume is not None:
        resume_path = last_checkpoint if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_record = load_training_state(
            resume_path, model, optimizer, scaler, train_loader, device
        )
        print(f"Resumed standard training from {resume_path}; next epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        lr = cosine_lr(args.lr, epoch - 1, args.epochs)
        set_lr(optimizer, lr)
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            args.label_smoothing,
        )
        validation = evaluate(model, validation_loader, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": lr,
            **train_metrics,
            "validation_examples": validation["examples"],
            "validation_loss": validation["loss"],
            "validation_top1": validation["top1"],
        }
        append_metrics(metrics_path, row)
        candidate = {
            "epoch": epoch,
            "validation_examples": validation["examples"],
            "validation_loss": validation["loss"],
            "validation_top1": validation["top1"],
        }
        if is_better(candidate, best_record):
            best_record = candidate
            atomic_torch_save(model.state_dict(), selected_weights)
        save_training_state(
            last_checkpoint, model, optimizer, scaler, train_loader, epoch,
            best_record, args,
        )
        atomic_torch_save(model.state_dict(), output_dir / "resnet20_standard_last_weights.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_training_state(
                output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                model, optimizer, scaler, train_loader, epoch, best_record, args,
            )
        print(
            f"standard epoch={epoch:03d} done | val Top1={float(validation['top1']):.3f}% "
            f"loss={float(validation['loss']):.4f} | best epoch={best_record['epoch']} "
            f"Top1={float(best_record['validation_top1']):.3f}%",
            flush=True,
        )

    if best_record is None or not selected_weights.is_file():
        raise RuntimeError("Standard training produced no selected checkpoint")
    model.load_state_dict(models.torch_load_compat(selected_weights, device))
    # This is the only official-test evaluation in the standard stage.
    test_metrics = evaluate(model, test_loader, device)
    final = {
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "PASS",
        "stage": "standard_reference",
        "selected_checkpoint": str(selected_weights),
        "selected_epoch": int(best_record["epoch"]),
        "selected_validation": best_record,
        "test_clean": test_metrics,
        "test_examples": int(test_metrics["examples"]),
        "official_full_test": not args.quick,
        "test_used_for_selection": False,
        "random_initialization": True,
        "training_examples": len(train_loader.dataset),
        "validation_examples": len(validation_loader.dataset),
        "split_file": str(args.split_file.expanduser().resolve()),
        "train_indices_sha256": split["train_indices_sha256"],
        "validation_indices_sha256": split["validation_indices_sha256"],
        "offline": True,
    }
    protocol.atomic_json(output_dir / "final_test_metrics.json", final)
    print("Standard reference training complete.")
    print("Selected:", selected_weights)
    print(
        f"Selected validation Top-1={float(best_record['validation_top1']):.3f}% "
        f"at epoch={best_record['epoch']}"
    )
    print(f"Official test Top-1={float(test_metrics['top1']):.3f}%")


if __name__ == "__main__":
    main()
