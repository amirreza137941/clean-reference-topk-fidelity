#!/usr/bin/env python3
"""Five preflight/self-tests for CIFAR-100 actual-defense calibration R1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import calibrate_topk_cifar100_actual_defense_r1 as evaluator
import topk_cifar100_defense_calibration_common as common


DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test CIFAR-100 defense calibration")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets/cifar-100-python"),
    )
    parser.add_argument("--r2-root", type=Path, default=DEFAULT_R2_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def test_pure_selection_rules() -> None:
    robust = {"top1_accuracy_percent": 60.0, "exact_set_probability_percent": 40.0}
    strengths = [
        {
            "strength_alpha_px": 0.5,
            "strength_steps": 1,
            "top1_accuracy_percent": 59.0,
            "exact_set_probability_percent": 50.0,
        },
        {
            "strength_alpha_px": 1.0,
            "strength_steps": 2,
            "top1_accuracy_percent": 57.0,
            "exact_set_probability_percent": 70.0,
        },
    ]
    selected, annotated = common.select_strength_candidate(strengths, robust, 2.0)
    assert selected["strength_alpha_px"] == 0.5
    assert [row["feasible"] for row in annotated] == [True, False]

    full0 = {"top1_accuracy_percent": 62.0, "exact_set_probability_percent": 55.0}
    sigma_rows = []
    for sigma, noise_top1, noise_exact, full_top1, full_exact in (
        (0.0, 60.0, 40.0, 62.0, 55.0),
        (0.01, 59.0, 39.0, 61.0, 54.0),
        (0.02, 57.0, 38.5, 60.0, 53.0),
    ):
        sigma_rows.extend(
            [
                {
                    "sigma": sigma,
                    "method": "noise_only",
                    "top1_accuracy_percent": noise_top1,
                    "exact_set_probability_percent": noise_exact,
                },
                {
                    "sigma": sigma,
                    "method": "full",
                    "top1_accuracy_percent": full_top1,
                    "exact_set_probability_percent": full_exact,
                },
            ]
        )
    selected_sigma, _ = common.select_sigma_candidate(sigma_rows, robust, full0, 2.0)
    assert selected_sigma["sigma"] == 0.01


def test_r2_integrity(project_dir: Path, r2_root: Path) -> None:
    manifest = common.read_json(r2_root / "pipeline_manifest.json")
    assert manifest["protocol_id"] == common.PARENT_PROTOCOL_ID
    for filename in (
        "topk_cifar100_r2_common.py",
        "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    ):
        expected = manifest["script_sha256"][filename]
        actual = common.file_sha256(project_dir / filename)
        if actual != expected:
            raise AssertionError(f"R2 source mismatch for {filename}: {actual} != {expected}")
    split = evaluator.load_frozen_split(r2_root / "frozen_split.json")
    assert split["train_indices_sha256"] == common.EXPECTED_TRAIN_HASH
    assert split["validation_indices_sha256"] == common.EXPECTED_VALIDATION_HASH

    for handoff_name in ("standard_to_downstream_handoff.json", "teacher_to_dual_handoff.json"):
        handoff = common.read_json(r2_root / handoff_name)
        checkpoint = Path(handoff["checkpoint"])
        if common.file_sha256(checkpoint) != handoff["checkpoint_sha256"]:
            raise AssertionError(f"R2 checkpoint mismatch: {checkpoint}")
    assert common.file_sha256(
        r2_root / "standard" / "resnet20_standard_best_validation_weights.pt"
    ) == common.EXPECTED_STANDARD_SHA256
    assert common.file_sha256(
        r2_root / "teacher_top2_at" / "resnet20_teacher_top2_best_composite_weights.pt"
    ) == common.EXPECTED_ANCHOR_SHA256


def test_data_partition(data_dir: Path, r2_root: Path) -> tuple[torch.utils.data.Dataset, list[int]]:
    root = evaluator.r2_anchor.resolve_torchvision_root(data_dir)
    dataset = torchvision.datasets.CIFAR100(
        root=str(root), train=True, download=False, transform=transforms.ToTensor()
    )
    split = evaluator.load_frozen_split(r2_root / "frozen_split.json")
    train, validation_from_common, _ = evaluator.r2_common.load_frozen_split(
        r2_root / "frozen_split.json", dataset.targets, seed=9103, val_size=5000
    )
    assert train == split["train_indices"]
    assert validation_from_common == split["validation_indices"]
    tune, confirm = evaluator.balanced_disjoint_subsets(
        dataset.targets, split["train_indices"], 1000, 1000, 2026090401
    )
    validation = evaluator.select_balanced_prefix(
        dataset.targets, split["validation_indices"], 5000
    )
    assert len(tune) == len(confirm) == 1000 and len(validation) == 5000
    assert not (set(tune) & set(confirm))
    assert not ((set(tune) | set(confirm)) & set(validation))
    assert set(np.bincount(np.asarray(dataset.targets)[tune], minlength=100)) == {10}
    assert set(np.bincount(np.asarray(dataset.targets)[confirm], minlength=100)) == {10}
    assert set(np.bincount(np.asarray(dataset.targets)[validation], minlength=100)) == {50}
    return dataset, validation[:4]


def test_models_and_strengthening(
    dataset: torch.utils.data.Dataset,
    indices: list[int],
    r2_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    standard_weights = r2_root / "standard" / "resnet20_standard_best_validation_weights.pt"
    anchor_weights = r2_root / "teacher_top2_at" / "resnet20_teacher_top2_best_composite_weights.pt"
    standard, anchor = evaluator.load_models(standard_weights, anchor_weights, device)
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    with torch.inference_mode():
        standard_logits = standard(images)
        anchor_logits = anchor(images)
    assert standard_logits.shape == anchor_logits.shape == (4, 100)
    assert torch.isfinite(standard_logits).all() and torch.isfinite(anchor_logits).all()
    reference = evaluator.ranked_top2(standard_logits)
    strengthened = evaluator.strengthen_inputs(
        images, reference, anchor, 1.275, 8.0, 2, 1.0, 4.0
    )
    assert strengthened.shape == images.shape
    assert float((strengthened - images).abs().max().item()) <= 8.0 / 255.0 + 1e-7
    assert float(strengthened.min().item()) >= 0.0 and float(strengthened.max().item()) <= 1.0
    return standard, anchor


def test_tiny_end_to_end(
    dataset: torch.utils.data.Dataset,
    indices: list[int],
    standard: torch.nn.Module,
    anchor: torch.nn.Module,
    device: torch.device,
) -> None:
    args = argparse.Namespace(
        batch_size=4,
        mc_chunk=2,
        num_workers=0,
        amp=False,
        top1_weight=1.0,
        set_weight=4.0,
    )
    loader = evaluator.make_loader(dataset, indices, args, device)
    methods = [
        evaluator.MethodSpec("standard", "standard"),
        evaluator.MethodSpec("robust", "anchor"),
        evaluator.MethodSpec("noise_only", "anchor", sigma=0.01, draws=4),
        evaluator.MethodSpec(
            "full",
            "anchor",
            strength_alpha_px=1.275,
            strength_epsilon_px=8.0,
            strength_steps=2,
            sigma=0.01,
            draws=4,
        ),
    ]
    summaries, rows = evaluator.evaluate_methods(
        "self_test", loader, methods, standard, anchor, device, args, 123456
    )
    assert len(summaries) == 4 and len(rows) == 16
    standard_rows = [row for row in rows if row["method"] == "standard"]
    assert all(row["exact_set_probability"] == 1.0 for row in standard_rows)
    stochastic_rows = [row for row in rows if row["method"] in {"noise_only", "full"}]
    assert all(abs(row["exact_set_probability"] * 4 - round(row["exact_set_probability"] * 4)) < 1e-9 for row in stochastic_rows)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    evaluator.seed_everything(2026090401)
    project_dir = Path(__file__).resolve().parent
    r2_root = args.r2_root.expanduser().resolve()

    test_pure_selection_rules()
    print("test_pure_selection_rules: OK")
    test_r2_integrity(project_dir, r2_root)
    print("test_r2_integrity: OK")
    dataset, indices = test_data_partition(args.data_dir, r2_root)
    print("test_data_partition: OK")
    standard, anchor = test_models_and_strengthening(dataset, indices, r2_root, device)
    print("test_models_and_strengthening: OK")
    test_tiny_end_to_end(dataset, indices, standard, anchor, device)
    print("test_tiny_end_to_end: OK")
    print("CIFAR-100 actual-defense calibration self-test: 5/5 OK")


if __name__ == "__main__":
    main()
