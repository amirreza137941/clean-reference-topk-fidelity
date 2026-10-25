#!/usr/bin/env python3
"""Pre-run self-test for CIFAR-100 official clean confirmation R3.

The integration test deliberately uses four training-set images.  It never
loads the official CIFAR-100 test split, so running this self-test does not add
an unregistered official-test evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import topk_cifar100_official_clean_r3_common as common


DEFAULT_DATA = Path("./datasets/cifar-100-python")
DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_DEV_DIR = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_dev_pareto_r2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test official clean R3")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--r2-root", type=Path, default=DEFAULT_R2_ROOT)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pure-only", action="store_true")
    return parser.parse_args()


def test_gate_logic() -> None:
    rows = []
    for spec in common.GATE_RULES:
        superiority = spec["rule"] == "ci95_low_pp_gt_zero"
        rows.append(
            {
                "comparison": spec["comparison"],
                "difference_pp": 1.0 if superiority else -0.5,
                "ci95_low_pp": 0.2 if superiority else -1.5,
                "ci95_high_pp": 1.8 if superiority else 0.5,
            }
        )
    decisions, ready = common.evaluate_gate_decisions(rows, 2.0)
    assert ready and all(row["passed"] for row in decisions)
    failing = [dict(row) for row in rows]
    target = common.GATE_RULES[0]["comparison"]
    next(row for row in failing if row["comparison"] == target)["ci95_low_pp"] = -2.0001
    decisions, ready = common.evaluate_gate_decisions(failing, 2.0)
    assert not ready and sum(not row["passed"] for row in decisions) == 1


def test_attack_partition() -> None:
    labels = np.repeat(np.arange(100, dtype=np.int64), 100)
    first = common.balanced_attack_partitions(labels, 2026090503)
    second = common.balanced_attack_partitions(labels, 2026090503)
    assert first == second
    assert len(first["pilot"]) == 200 and len(set(first["pilot"])) == 200
    assert len(first["confirmatory"]) == 1000 and len(set(first["confirmatory"])) == 1000
    assert not set(first["pilot"]) & set(first["confirmatory"])
    for key, per_class in (("pilot", 2), ("confirmatory", 10)):
        counts = np.bincount(labels[first[key]], minlength=100)
        assert np.all(counts == per_class)


def test_bootstrap_reproducibility() -> None:
    delta = np.asarray([0.0, 0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    specs = [{"comparison": "synthetic", "left": "a", "right": "b", "metric": "top1", "delta": delta}]
    first = common.paired_bootstrap_many(specs, 1000, 17, chunk_size=13)
    second = common.paired_bootstrap_many(specs, 1000, 17, chunk_size=13)
    assert first == second
    assert abs(first[0]["difference_pp"] - 100.0 * float(delta.mean())) < 1e-12


def test_parent_selection(args: argparse.Namespace) -> None:
    selection = common.read_json(args.dev_dir / "dev_pareto_selection.json")
    final = common.read_json(args.dev_dir / "final_summary.json")
    config = common.read_json(args.dev_dir / "run_config.json")
    common.validate_frozen_selection(selection, final, config)
    payload = dict(config)
    claimed = str(payload.pop("config_hash"))
    assert common.object_sha256(payload) == claimed == common.EXPECTED_DEV_CONFIG_HASH


def test_parent_sources_and_weights(args: argparse.Namespace) -> None:
    import evaluate_topk_cifar100_official_clean_r3 as evaluator

    parents = evaluator.load_and_validate_parents(args)
    assert common.file_sha256(parents["paths"]["standard_weights"]) == common.EXPECTED_STANDARD_SHA256
    assert common.file_sha256(parents["paths"]["anchor_weights"]) == common.EXPECTED_ANCHOR_SHA256
    methods = evaluator.selected_methods(parents["selection"], 100)
    assert tuple(method.name for method in methods) == common.METHODS
    assert methods[-1].sigma == 0.015 and methods[-1].strength_steps == 2


def test_train_side_tiny_end_to_end(args: argparse.Namespace) -> None:
    import torch
    import torchvision
    import torchvision.transforms as transforms

    import evaluate_topk_cifar100_official_clean_r3 as evaluator

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    evaluator.seed_everything(2026090501)
    parents = evaluator.load_and_validate_parents(args)
    dataset_root = evaluator.dev.r1.r2_anchor.resolve_torchvision_root(args.data_dir)
    # train=True is intentional: the self-test must not read the official test.
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root), train=True, download=False, transform=transforms.ToTensor()
    )
    tiny_args = SimpleNamespace(
        batch_size=4,
        num_workers=0,
        mc_chunk=2,
        amp=False,
        top1_weight=1.0,
        set_weight=4.0,
    )
    standard, anchor = evaluator.dev.r1.load_models(
        parents["paths"]["standard_weights"],
        parents["paths"]["anchor_weights"],
        device,
    )
    methods = evaluator.selected_methods(parents["selection"], 4)
    loader = evaluator.dev.make_loader(dataset, [0, 1, 2, 3], tiny_args, device)
    summaries, rows = evaluator.dev.evaluate_methods(
        "train_side_self_test",
        loader,
        methods,
        standard,
        anchor,
        device,
        tiny_args,
        evaluator.dev.stable_seed(2026090501, "train_side_self_test"),
    )
    assert len(summaries) == 5 and len(rows) == 20
    for method in common.METHODS:
        selected = [row for row in rows if row["method"] == method]
        assert len(selected) == 4
    standard_rows = [row for row in rows if row["method"] == "standard"]
    assert all(row["exact_set_probability"] == 1.0 for row in standard_rows)


def main() -> None:
    args = parse_args()
    test_gate_logic()
    print("test_gate_logic: OK")
    test_attack_partition()
    print("test_attack_partition: OK")
    test_bootstrap_reproducibility()
    print("test_bootstrap_reproducibility: OK")
    if args.pure_only:
        print("CIFAR-100 official-clean R3 pure self-test: 3/3 OK")
        return
    test_parent_selection(args)
    print("test_parent_selection: OK")
    test_parent_sources_and_weights(args)
    print("test_parent_sources_and_weights: OK")
    test_train_side_tiny_end_to_end(args)
    print("test_train_side_tiny_end_to_end: OK (train=True; official test not read)")
    print("CIFAR-100 official-clean R3 self-test: 6/6 OK")


if __name__ == "__main__":
    main()
