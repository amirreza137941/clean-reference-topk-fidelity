#!/usr/bin/env python3
"""Pre-run self-tests for the frozen ImageNet R3 protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_topk_imagenet_clean_confirmation_r3 as evaluator
import topk_imagenet_clean_confirmation_r3_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test ImageNet R3 clean confirmation")
    parser.add_argument("--data-root", type=Path, default=evaluator.DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=evaluator.DEFAULT_R0_DIR)
    parser.add_argument("--r1-dir", type=Path, default=evaluator.DEFAULT_R1_DIR)
    parser.add_argument("--r2-dir", type=Path, default=evaluator.DEFAULT_R2_DIR)
    parser.add_argument(
        "--standard-checkpoint", type=Path, default=evaluator.DEFAULT_STANDARD_CHECKPOINT
    )
    parser.add_argument(
        "--robust-checkpoint", type=Path, default=evaluator.DEFAULT_ROBUST_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def test_frozen_protocol() -> None:
    protocol = common.frozen_protocol_payload()
    claimed = protocol.pop("protocol_hash")
    assert claimed == common.object_sha256(protocol)
    assert common.PARTITION == "clean_confirmation"
    assert common.EXAMPLES == 45_000
    assert common.EXAMPLES_PER_CLASS == 45
    assert common.SHARDS * common.EXAMPLES_PER_SHARD == common.EXAMPLES
    assert common.PRIMARY_K == 5 and common.SECONDARY_K == 2
    assert common.MC_DRAWS == 50 and common.SIGMA == 0.025
    assert common.FROZEN_STRENGTH == {
        "candidate_id": "global_a1_s4_w4",
        "gate_mode": "global",
        "alpha_px": 1.0,
        "epsilon_px": 4.0,
        "steps": 4,
        "top1_weight": 1.0,
        "set_weight": 4.0,
    }
    assert len(common.GATE_DEFINITIONS) == 10


def test_pure_statistics() -> None:
    class_delta = np.linspace(-0.02, 0.03, common.CLASSES)
    specs = [
        {
            "protocol_id": common.PROTOCOL_ID,
            "comparison_id": "synthetic",
            "comparison": "left - right",
            "left": "left",
            "right": "right",
            "metric": "top1",
            "metric_key": common.METRIC_KEYS["top1"],
            "examples": common.EXAMPLES,
            "class_delta": class_delta,
        }
    ]
    first = common.paired_class_bootstrap_many(specs, 1000, 17, chunk_size=13)
    second = common.paired_class_bootstrap_many(specs, 1000, 17, chunk_size=13)
    assert first == second
    assert abs(first[0]["difference_pp"] - 0.5) < 1e-12
    assert first[0]["clusters"] == 1000

    comparison_rows = []
    for _, comparison_id, kind in common.GATE_DEFINITIONS:
        comparison_rows.append(
            {
                "comparison_id": comparison_id,
                "difference_pp": 1.0,
                "ci95_low_pp": -1.0 if kind == "noninferiority" else 0.1,
                "ci95_high_pp": 2.0,
            }
        )
    gates = common.apply_preregistered_gates(comparison_rows)
    assert len(gates) == 10 and all(row["passed"] for row in gates)
    comparison_rows[0]["ci95_low_pp"] = -2.001
    assert common.apply_preregistered_gates(comparison_rows)[0]["passed"] is False


def parent_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "data_root": evaluator.resolved(args.data_root),
        "r0_dir": evaluator.resolved(args.r0_dir),
        "r1_dir": evaluator.resolved(args.r1_dir),
        "r2_dir": evaluator.resolved(args.r2_dir),
        "standard_checkpoint": evaluator.resolved(args.standard_checkpoint),
        "robust_checkpoint": evaluator.resolved(args.robust_checkpoint),
        "output_dir": Path("/unused/by/self-test"),
    }


def test_parent_and_partition(args: argparse.Namespace) -> None:
    paths = parent_paths(args)
    rows, parents = evaluator.validate_parents(paths)
    assert len(rows) == common.EXAMPLES
    assert parents["r2_final_summary"]["official_clean_stage_ready"] is True
    shards = evaluator.outcome_blind_shards(rows)
    assert len(shards) == common.SHARDS
    assert {len(shard) for shard in shards} == {common.EXAMPLES_PER_SHARD}
    assert len({row["relative_path"] for shard in shards for row in shard}) == common.EXAMPLES
    counts = np.bincount(
        np.asarray([int(row["target"]) for row in rows]), minlength=common.CLASSES
    )
    assert np.array_equal(counts, np.full(common.CLASSES, common.EXAMPLES_PER_CLASS))
    # The self-test validates file existence but deliberately opens no holdout image.
    for row in rows:
        assert (paths["data_root"] / str(row["relative_path"])).is_file()


def test_models_and_frozen_defense(args: argparse.Namespace) -> None:
    runtime_args = SimpleNamespace(
        device=args.device,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )
    device = evaluator.configure_device(runtime_args)
    evaluator.seed_everything(common.EVALUATION_SEED)
    standard, robust, audit = evaluator.load_models(
        evaluator.resolved(args.standard_checkpoint),
        evaluator.resolved(args.robust_checkpoint),
        device,
    )
    assert audit["robust"]["state_dict_fingerprint_sha256"] == common.ROBUST_STATE_FINGERPRINT
    generator = torch.Generator(device=device)
    generator.manual_seed(314159)
    images = torch.rand((2, 3, 224, 224), generator=generator, device=device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        standard_logits = standard(images).float()
        robust_logits = robust(images).float()
        reference = evaluator.r2.ranked_reference(standard_logits, common.PRIMARY_K).clone()
        disagreement = standard_logits.argmax(1).ne(robust_logits.argmax(1))
    base, active = evaluator.r2.selected_strength_base(
        images,
        reference,
        disagreement,
        robust,
        evaluator.frozen_strength_spec(),
        False,
    )
    assert base.shape == images.shape and bool(active.all().item())
    assert float((base - images).abs().max().item()) <= 4.0 / 255.0 + 1e-6
    with torch.no_grad():
        logits = robust(base).float()
    metrics = evaluator.r2.metric_tensors(logits, labels, reference)
    assert set(metrics) == {
        "top1_accuracy",
        "reference_top1_match",
        "label_coverage_k5",
        "exact_set_k5",
        "label_coverage_k2",
        "exact_set_k2",
    }


def test_row_schema() -> None:
    labels = torch.tensor([1, 2])
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    reference = torch.tensor([[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]])
    rows = evaluator.deterministic_rows(
        "shard_00",
        evaluator.DefenseSpec("standard"),
        logits,
        labels,
        ["a.jpeg", "b.jpeg"],
        reference,
    )
    assert len(rows) == 2
    assert all(row["protocol_id"] == common.PROTOCOL_ID for row in rows)
    assert all(row["exact_set_k5_probability"] == 1.0 for row in rows)
    assert all(row["draws"] == 1 for row in rows)


def main() -> None:
    args = parse_args()
    test_frozen_protocol()
    print("test_frozen_protocol: OK")
    test_pure_statistics()
    print("test_pure_statistics: OK")
    test_parent_and_partition(args)
    print("test_parent_and_partition: OK (no holdout pixels opened)")
    test_models_and_frozen_defense(args)
    print("test_models_and_frozen_defense: OK")
    test_row_schema()
    print("test_row_schema: OK")
    print("All five ImageNet R3 self-tests passed.")


if __name__ == "__main__":
    main()
