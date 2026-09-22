#!/usr/bin/env python3
"""Offline preflight and immutable-split creation for CIFAR-100 R2."""

from __future__ import annotations

import argparse
import platform
from collections import Counter
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

import topk_cifar100_r2_common as protocol
import train_resnet20_teacher_top2_adversarial_cifar100_r2 as models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local CIFAR-100 and freeze the leakage-free R2 split"
    )
    parser.add_argument("--data-dir", type=Path, default=protocol.DEFAULT_DATA)
    parser.add_argument(
        "--output-dir", type=Path, default=protocol.DEFAULT_ROOT / "preflight"
    )
    parser.add_argument(
        "--split-file", type=Path,
        default=protocol.DEFAULT_ROOT / "frozen_split.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-seed", type=int, default=protocol.DEFAULT_SEED)
    parser.add_argument("--val-size", type=int, default=protocol.DEFAULT_VAL_SIZE)
    return parser.parse_args()


def class_histogram(targets) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(int(x) for x in targets).items())
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    root = protocol.resolve_torchvision_root(data_dir)
    archive = root / "cifar-100-python"
    files = {name: archive / name for name in ("train", "test", "meta")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CIFAR-100 files missing: {missing}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    plain = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR100(
        root=str(root), train=True, download=False, transform=plain
    )
    test_set = torchvision.datasets.CIFAR100(
        root=str(root), train=False, download=False, transform=plain
    )
    if len(train_set) != 50_000 or len(test_set) != 10_000:
        raise AssertionError(
            f"Unexpected CIFAR-100 sizes: train={len(train_set)}, test={len(test_set)}"
        )
    if set(Counter(train_set.targets).values()) != {500}:
        raise AssertionError("CIFAR-100 training data is not 100 x 500 balanced")
    if set(Counter(test_set.targets).values()) != {100}:
        raise AssertionError("CIFAR-100 test data is not 100 x 100 balanced")

    split = protocol.create_or_validate_frozen_split(
        split_file, train_set.targets, args.split_seed, args.val_size
    )
    # Architecture-only smoke test. R2 intentionally has no incoming checkpoint.
    model = models.make_model().to(device).eval()
    images = torch.stack([train_set[int(i)][0] for i in split["validation_indices"][:4]])
    with torch.inference_mode():
        logits = model(images.to(device))
    if tuple(logits.shape) != (4, 100) or not torch.isfinite(logits).all():
        raise AssertionError(f"ResNet-20 smoke forward failed: {tuple(logits.shape)}")

    report = {
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "PASS",
        "offline": True,
        "downloads_attempted": False,
        "incoming_checkpoint_used": False,
        "dataset": "CIFAR-100",
        "data_dir": str(data_dir),
        "torchvision_root": str(root),
        "dataset_files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": protocol.sha256_file(path),
            }
            for name, path in files.items()
        },
        "train_examples": len(train_set),
        "test_examples": len(test_set),
        "train_class_histogram": class_histogram(train_set.targets),
        "test_class_histogram": class_histogram(test_set.targets),
        "split_file": str(split_file),
        "split_seed": args.split_seed,
        "training_examples": split["training_size"],
        "validation_examples": split["validation_size"],
        "train_indices_sha256": split["train_indices_sha256"],
        "validation_indices_sha256": split["validation_indices_sha256"],
        "labels_sha256": split["labels_sha256"],
        "normalization": {
            "mean": list(protocol.CIFAR100_MEAN),
            "std": list(protocol.CIFAR100_STD),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    protocol.atomic_json(output_dir / "preflight_report.json", report)
    print(
        "PASS CIFAR-100 R2 preflight: 45000 train / 5000 validation / "
        "10000 untouched official test"
    )
    print(f"Frozen split: {split_file}")
    print(f"Validation SHA256: {split['validation_indices_sha256']}")
    print("No pretrained checkpoint is consumed by the R2 protocol.")


if __name__ == "__main__":
    main()
