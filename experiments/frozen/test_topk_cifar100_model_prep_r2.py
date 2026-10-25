#!/usr/bin/env python3
"""Fast server-side self-tests for the leakage-free CIFAR-100 R2 package."""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
import torchvision.transforms as transforms

import topk_cifar100_r2_common as protocol
import train_resnet20_dual_backbone_top2_cifar100_r2 as dual
import train_resnet20_teacher_top2_adversarial_cifar100_r2 as anchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test CIFAR-100 R2")
    parser.add_argument("--data-dir", type=Path, default=protocol.DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def test_offline_sources() -> None:
    project = Path(__file__).resolve().parent
    names = [
        "topk_cifar100_r2_common.py",
        "preflight_topk_cifar100_model_prep_r2.py",
        "train_resnet20_standard_cifar100_r2.py",
        "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
        "train_resnet20_dual_backbone_top2_cifar100_r2.py",
        "run_topk_cifar100_model_prep_r2.py",
    ]
    for name in names:
        source = (project / name).read_text(encoding="utf-8")
        if "download=True" in source.replace(" ", ""):
            raise AssertionError(f"{name} enables a dataset download")
        for forbidden in ("requests.", "urllib.request", "torch.hub", "wget ", "curl "):
            if forbidden in source:
                raise AssertionError(f"{name} contains network access: {forbidden}")


def load_dataset(data_dir: Path):
    root = protocol.resolve_torchvision_root(data_dir)
    dataset = torchvision.datasets.CIFAR100(
        root=str(root), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    if len(dataset) != 50_000 or set(Counter(dataset.targets).values()) != {500}:
        raise AssertionError("Local CIFAR-100 training set is incomplete/unbalanced")
    return dataset


def test_split(dataset) -> list[int]:
    with tempfile.TemporaryDirectory(prefix="topk-cifar100-r2-test-") as directory:
        path = Path(directory) / "frozen_split.json"
        first = protocol.create_or_validate_frozen_split(
            path, dataset.targets, protocol.DEFAULT_SEED, 5000
        )
        second = protocol.create_or_validate_frozen_split(
            path, dataset.targets, protocol.DEFAULT_SEED, 5000
        )
        if first != second:
            raise AssertionError("Frozen split is not reproducible")
        train_indices, validation_indices, _ = protocol.load_frozen_split(
            path, dataset.targets, protocol.DEFAULT_SEED, 5000
        )
        if len(train_indices) != 45_000 or len(validation_indices) != 5_000:
            raise AssertionError("Frozen split sizes are not 45k/5k")
        if set(train_indices).intersection(validation_indices):
            raise AssertionError("Frozen split overlaps")
        counts = Counter(dataset.targets[index] for index in validation_indices)
        if len(counts) != 100 or set(counts.values()) != {50}:
            raise AssertionError("Frozen validation is not 50 examples/class")
        return validation_indices[:8]


def test_architecture_and_composition(dataset, indices: list[int], device: torch.device) -> None:
    images = torch.stack([dataset[index][0] for index in indices[:4]]).to(device)
    reference = anchor.make_model().to(device).eval()
    defense = dual.DualBackboneDefense().to(device).eval()
    with torch.inference_mode():
        reference_logits = reference(images)
        robust_logits, _, final_logits = defense.forward_all(images)
    if tuple(reference_logits.shape) != (4, 100):
        raise AssertionError("ResNet-20 does not output 100 logits")
    if not torch.equal(robust_logits.argmax(1), final_logits.argmax(1)):
        raise AssertionError("Dual composition does not preserve robust Top-1")
    if not torch.isfinite(final_logits).all():
        raise AssertionError("Dual composition produced non-finite logits")


def test_attacks(dataset, indices: list[int], device: torch.device) -> None:
    images = torch.stack([dataset[index][0] for index in indices[:4]]).to(device)
    labels = torch.tensor([dataset.targets[index] for index in indices[:4]], device=device)
    reference = anchor.make_model().to(device).eval()
    with torch.inference_mode():
        teacher_top2 = anchor.unordered_top2(reference(images))
    adversarial = anchor.joint_pgd_attack(
        reference, images, labels, teacher_top2,
        epsilon=2.0 / 255.0, alpha=0.5 / 255.0, steps=1,
        top1_weight=0.5, set2_weight=0.5, set2_margin=0.5,
    )
    if (adversarial - images).abs().max().item() > 2.0 / 255.0 + 1e-6:
        raise AssertionError("Anchor PGD exceeded the 2/255 Linf radius")
    defense = dual.DualBackboneDefense().to(device).eval()
    attack_args = SimpleNamespace(
        set2_margin=0.5, attack_top1_weight=0.25, attack_set2_weight=0.75
    )
    attacked_dual = dual.pgd_attack(
        defense, images, labels, teacher_top2,
        epsilon=2.0 / 255.0, alpha=0.5 / 255.0, steps=1,
        args=attack_args, goal="joint",
    )
    if (attacked_dual - images).abs().max().item() > 2.0 / 255.0 + 1e-6:
        raise AssertionError("Dual PGD exceeded the 2/255 Linf radius")


def test_frozen_defaults() -> None:
    if anchor.epsilon_for_epoch(1, 2.0 / 255.0, 5) != 0.0:
        raise AssertionError("Anchor warmup must start at epsilon zero")
    if anchor.epsilon_for_epoch(5, 2.0 / 255.0, 5) != 2.0 / 255.0:
        raise AssertionError("Anchor warmup must reach 2/255 at epoch 5")
    if dual.epsilon_for_epoch(20, 2.0 / 255.0, 20) != 2.0 / 255.0:
        raise AssertionError("Dual warmup must reach 2/255 at epoch 20")
    if dual.teacher_multihot(torch.tensor([[0, 99]])).shape != (1, 100):
        raise AssertionError("Dual target is not dimensioned for 100 classes")
    source = (
        Path(__file__).resolve().parent
        / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py"
    ).read_text(encoding="utf-8")
    required = (
        '"selection_eligible": selection_eligible',
        "current_epsilon >= args.epsilon - 1e-12",
        "No full-radius selectable weights were saved",
    )
    for marker in required:
        if marker not in source:
            raise AssertionError(f"Anchor full-radius selection guard missing: {marker}")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    test_offline_sources()
    print("PASS 1/5: package has no download/network path")
    dataset = load_dataset(args.data_dir.expanduser().resolve())
    print("PASS 2/5: local CIFAR-100 is complete")
    indices = test_split(dataset)
    print("PASS 3/5: immutable 45k/5k split is deterministic and disjoint")
    test_architecture_and_composition(dataset, indices, device)
    print("PASS 4/5: 100-class models and Dual Top-1 composition")
    test_attacks(dataset, indices, device)
    test_frozen_defaults()
    print("PASS 5/5: 2/255 attacks, warmup, and full-radius selection guards")
    print("CIFAR-100 R2 self-test complete.")


if __name__ == "__main__":
    main()
