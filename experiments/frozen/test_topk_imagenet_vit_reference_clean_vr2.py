#!/usr/bin/env python3
"""Self-tests for ImageNet ViT-reference clean confirmation VR2."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import topk_imagenet_vit_reference_clean_vr2_common as common


DEFAULT_ROOT = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test VR2 clean confirmation")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT / "datasets/imagenet")
    parser.add_argument(
        "--r0-dir",
        type=Path,
        default=DEFAULT_ROOT / "models/imagenet_topk/paper_topk_imagenet_preflight_r0",
    )
    parser.add_argument(
        "--r1-dir",
        type=Path,
        default=DEFAULT_ROOT / "models/imagenet_topk/paper_topk_imagenet_robust_anchor_r1",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "models/imagenet_topk/paper_topk_imagenet_vit_reference_crossarch_calibration_dev_vr1"
        ),
    )
    parser.add_argument(
        "--standard-checkpoint",
        type=Path,
        default=DEFAULT_ROOT / "models/vit/imagenet/vit_b16_imagenet.pth",
    )
    parser.add_argument(
        "--robust-checkpoint",
        type=Path,
        default=DEFAULT_ROOT / "models/resnet50/imagenet/salman2020do_r50_linf4.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--pure-only", action="store_true")
    return parser.parse_args()


def test_constants() -> None:
    assert common.CLEAN_EXAMPLES == 45_000
    assert common.CLEAN_EXAMPLES_PER_CLASS == 45
    assert common.EXAMPLES_PER_SHARD == 500
    assert common.CLEAN_SHARDS == 90
    assert common.MC_DRAWS == 50
    assert common.BOOTSTRAP_RESAMPLES == 50_000
    assert common.shard_id(0) == "shard_000"
    assert common.shard_id(89) == "shard_089"
    try:
        common.shard_id(90)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range shard ID was accepted")


def _bootstrap_rows(default_low: float = 0.25) -> list[dict[str, object]]:
    rows = []
    for name, (rule, _threshold) in common.GATE_RULES.items():
        low = default_low if rule == "superiority" else -1.0
        rows.append(
            {
                "comparison": name,
                "difference_pp": low + 0.5,
                "ci95_low_pp": low,
                "ci95_high_pp": low + 1.0,
            }
        )
    return rows


def test_gate() -> None:
    passed = common.evaluate_clean_gate(_bootstrap_rows())
    assert passed["passed"] is True
    assert passed["blockers"] == []
    failed_rows = _bootstrap_rows()
    for row in failed_rows:
        if row["comparison"] == "full_minus_robust_exact_k5":
            row["ci95_low_pp"] = 0.0
    failed = common.evaluate_clean_gate(failed_rows)
    assert failed["passed"] is False
    assert len(failed["blockers"]) == 1
    ni_failed_rows = _bootstrap_rows()
    for row in ni_failed_rows:
        if row["comparison"] == "noise_only_minus_robust_top1":
            row["ci95_low_pp"] = -2.0001
    assert common.evaluate_clean_gate(ni_failed_rows)["passed"] is False


def test_frozen_protocol() -> None:
    payload = common.frozen_protocol_payload()
    claimed = payload["protocol_hash"]
    unhashed = dict(payload)
    del unhashed["protocol_hash"]
    assert common.object_sha256(unhashed) == claimed
    assert len(payload["gate_rules"]) == 10
    assert payload["bootstrap"] == {
        "unit": "ImageNet class cluster",
        "clusters": 1_000,
        "resamples": 50_000,
        "seed": common.BOOTSTRAP_SEED,
        "interval": "paired percentile 95%",
    }


def test_operating_point_contract() -> None:
    selection = {
        "selection_hash": common.EXPECTED_SELECTION_HASH,
        "config_hash": common.EXPECTED_CALIBRATION_CONFIG_HASH,
        "selection_split": "calibration_dev",
        "strength": {
            "candidate_id": "global_a1_s4_w4",
            "gate_mode": "global",
            "alpha_px": 1.0,
            "epsilon_px": 4.0,
            "steps": 4,
            "top1_weight": 1.0,
            "set_weight": 4.0,
            "positive_strength_selected": True,
            "selected_dev_summary": {"jointly_ci_feasible": True},
        },
        "randomization": {
            "sigma": 0.025,
            "draws_during_calibration": 50,
            "positive_sigma_selected": True,
            "selected_dev_constraints": {"jointly_ci_feasible": True},
        },
        "data_governance": {
            "partitions_read": ["calibration_dev"],
            "attack_outcomes_read": False,
            "training_performed": False,
            "cifar_selected_hyperparameters_transferred_as_fixed_values": False,
            "claim_scope": "cross_architecture_reference_only_not_robust_vit",
        },
    }
    common.validate_frozen_operating_point(selection)
    bad = {**selection, "randomization": {"sigma": 0.0275, "draws_during_calibration": 50}}
    try:
        common.validate_frozen_operating_point(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("A retuned sigma was accepted")


def run_pure_tests() -> None:
    test_constants()
    print("test_constants: OK")
    test_gate()
    print("test_gate: OK")
    test_frozen_protocol()
    print("test_frozen_protocol: OK")
    test_operating_point_contract()
    print("test_operating_point_contract: OK")


def run_full_tests(args: argparse.Namespace) -> None:
    import torch

    import confirm_topk_imagenet_vit_reference_clean_vr2 as evaluator

    paths = {
        "data_root": evaluator.resolved(args.data_root),
        "r0_dir": evaluator.resolved(args.r0_dir),
        "r1_dir": evaluator.resolved(args.r1_dir),
        "calibration_dir": evaluator.resolved(args.calibration_dir),
        "standard_checkpoint": evaluator.resolved(args.standard_checkpoint),
        "robust_checkpoint": evaluator.resolved(args.robust_checkpoint),
    }
    clean_rows, parents = evaluator.validate_calibration_parent(paths)
    assert len(clean_rows) == common.CLEAN_EXAMPLES
    assert parents["selection"]["selection_hash"] == common.EXPECTED_SELECTION_HASH
    print("test_parents_and_clean_partition: OK")

    shards = evaluator.outcome_blind_shards(clean_rows)
    assert len(shards) == common.CLEAN_SHARDS
    assert len({str(row["relative_path"]) for shard in shards for row in shard}) == 45_000
    print("test_outcome_blind_shards: OK")

    device_args = SimpleNamespace(
        device=args.device, gpu_memory_fraction=float(args.gpu_memory_fraction)
    )
    device = evaluator.calibration.configure_device(device_args)
    evaluator.calibration.seed_everything(2026091791)
    standard, robust, _audit = evaluator.calibration.load_models(
        paths["standard_checkpoint"], paths["robust_checkpoint"], device
    )
    tiny_rows = shards[0][:2]
    loader = evaluator.calibration.make_loader(
        paths["data_root"], tiny_rows, batch_size=2, num_workers=0, device=device
    )
    tiny_args = SimpleNamespace(mc_draws=2, mc_chunk=1, seed=2026091792)
    rows = evaluator.evaluate_clean_shard(
        "shard_000",
        loader,
        standard,
        robust,
        parents["selection"],
        device,
        tiny_args,
    )
    assert len(rows) == 2 * len(common.METHODS)
    assert {str(row["method"]) for row in rows} == set(common.METHODS)
    assert all(row["protocol_id"] == common.PROTOCOL_ID for row in rows)
    standard_rows = [row for row in rows if row["method"] == "standard"]
    assert all(float(row["exact_set_k5_probability"]) == 1.0 for row in standard_rows)
    full_rows = [row for row in rows if row["method"] in {"full", "full_sigma0"}]
    assert all(int(row["gate_active"]) == 1 for row in full_rows)
    for row in rows:
        for key in common.PROBABILITY_METRICS.values():
            assert 0.0 <= float(row[key]) <= 1.0
    print("test_tiny_official_clean_path: OK")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    run_pure_tests()
    if not args.pure_only:
        run_full_tests(args)
    print("SELF_TEST_STATUS=PASS")


if __name__ == "__main__":
    main()
