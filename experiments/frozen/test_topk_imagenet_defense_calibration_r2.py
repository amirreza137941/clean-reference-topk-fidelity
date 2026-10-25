#!/usr/bin/env python3
"""Seven self-tests for ImageNet Top-k defense calibration R2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import calibrate_topk_imagenet_defense_r2 as evaluator
import topk_imagenet_defense_calibration_r2_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test ImageNet calibration R2")
    parser.add_argument("--data-root", type=Path, default=evaluator.DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=evaluator.DEFAULT_R0_DIR)
    parser.add_argument("--r1-dir", type=Path, default=evaluator.DEFAULT_R1_DIR)
    parser.add_argument(
        "--standard-checkpoint", type=Path, default=evaluator.DEFAULT_STANDARD_CHECKPOINT
    )
    parser.add_argument(
        "--robust-checkpoint", type=Path, default=evaluator.DEFAULT_ROBUST_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def test_frozen_grid() -> None:
    assert common.PRIMARY_K == 5 and common.SECONDARY_K == 2
    assert len(common.STRENGTH_GRID) == 27
    assert len(common.GATE_MODES) == 2
    assert len(common.SIGMA_GRID) == 13
    assert common.SIGMA_GRID[0] == 0.0 and common.SIGMA_GRID[-1] == 0.03
    assert {row["epsilon_px"] for row in common.STRENGTH_GRID} == {4.0}
    assert {row["alpha_px"] for row in common.STRENGTH_GRID} == {0.25, 0.5, 1.0}
    assert {row["steps"] for row in common.STRENGTH_GRID} == {1, 2, 4}
    assert {row["set_weight"] for row in common.STRENGTH_GRID} == {1.0, 2.0, 4.0}
    candidate_ids = {
        common.strength_candidate_id(gate, candidate)
        for candidate in common.STRENGTH_GRID
        for gate in common.GATE_MODES
    }
    assert len(candidate_ids) == 54


def test_pure_selection() -> None:
    candidates = [
        {
            "candidate_id": "identity",
            "gate_mode": "none",
            "strength_steps": 0,
            "strength_alpha_px": 0.0,
            "set_weight": 0.0,
            "top1_accuracy_percent": 64.0,
            "label_coverage_k5_probability_percent": 84.0,
            "exact_set_k5_probability_percent": 25.0,
        },
        {
            "candidate_id": "bad_top5",
            "gate_mode": "global",
            "strength_steps": 2,
            "strength_alpha_px": 0.5,
            "set_weight": 4.0,
            "top1_accuracy_percent": 64.0,
            "label_coverage_k5_probability_percent": 80.0,
            "exact_set_k5_probability_percent": 90.0,
        },
        {
            "candidate_id": "feasible",
            "gate_mode": "disagreement",
            "strength_steps": 2,
            "strength_alpha_px": 0.5,
            "set_weight": 2.0,
            "top1_accuracy_percent": 63.8,
            "label_coverage_k5_probability_percent": 83.8,
            "exact_set_k5_probability_percent": 70.0,
        },
    ]
    bootstrap = []
    for candidate in candidates:
        for metric in common.STRENGTH_CONSTRAINTS:
            low = 0.0 if candidate["candidate_id"] == "identity" else -1.0
            if candidate["candidate_id"] == "bad_top5" and metric == "top5":
                low = -2.1
            bootstrap.append(
                {
                    "group_id": candidate["candidate_id"],
                    "metric": metric,
                    "difference_pp": low + 0.2,
                    "ci95_low_pp": low,
                    "ci95_high_pp": low + 0.4,
                }
            )
    selected, annotated = common.select_strength_candidate(candidates, bootstrap, 2.0)
    assert selected["candidate_id"] == "feasible"
    assert [row["jointly_ci_feasible"] for row in annotated] == [True, False, True]

    sigma_rows = [{"sigma": sigma} for sigma in (0.0, 0.01, 0.02)]
    sigma_bootstrap = []
    for sigma in (0.0, 0.01, 0.02):
        group_id = "sigma_" + common.sigma_tag(sigma)
        for constraint in common.SIGMA_CONSTRAINTS:
            low = 0.0 if sigma == 0.0 else -1.0
            if sigma == 0.02 and constraint == "full_top5_vs_robust":
                low = -2.01
            sigma_bootstrap.append(
                {
                    "group_id": group_id,
                    "metric": constraint,
                    "difference_pp": low + 0.1,
                    "ci95_low_pp": low,
                    "ci95_high_pp": low + 0.2,
                }
            )
    selected_sigma, _ = common.select_sigma_candidate(sigma_rows, sigma_bootstrap, 2.0)
    assert selected_sigma["sigma"] == 0.01


def test_bootstrap() -> None:
    delta = np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float64)
    spec = [{"group_id": "x", "metric": "top1", "delta": delta}]
    first = common.paired_bootstrap_many(spec, 1000, 19, chunk_size=7)
    second = common.paired_bootstrap_many(spec, 1000, 19, chunk_size=7)
    assert first == second
    assert abs(first[0]["difference_pp"] - 12.5) < 1e-12


def test_parent_and_partition(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[list[dict[str, object]]]]:
    paths = {
        "data_root": evaluator.resolved(args.data_root),
        "r0_dir": evaluator.resolved(args.r0_dir),
        "r1_dir": evaluator.resolved(args.r1_dir),
        "standard_checkpoint": evaluator.resolved(args.standard_checkpoint),
        "robust_checkpoint": evaluator.resolved(args.robust_checkpoint),
    }
    rows, _ = evaluator.validate_parents(paths)
    shards = evaluator.outcome_blind_shards(rows)
    assert len(rows) == 2000 and len(shards) == 20
    assert all(len(shard) == 100 for shard in shards)
    assert len({row["relative_path"] for shard in shards for row in shard}) == 2000
    for row in rows:
        path = paths["data_root"] / str(row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
    return rows, shards


def test_models_and_metrics(
    args: argparse.Namespace,
    probe_rows: list[dict[str, object]],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    standard, robust, audits = evaluator.load_models(
        evaluator.resolved(args.standard_checkpoint),
        evaluator.resolved(args.robust_checkpoint),
        device,
    )
    assert audits["robust"]["state_dict_fingerprint_sha256"] == common.ROBUST_STATE_FINGERPRINT
    loader = evaluator.make_loader(
        evaluator.resolved(args.data_root), probe_rows, 8, 0, device
    )
    images, labels, relative_paths = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    paths = [str(value) for value in relative_paths]
    with torch.no_grad():
        standard_logits = standard(images).float()
        robust_logits = robust(images).float()
        reference = evaluator.ranked_reference(standard_logits, 5).clone()
    assert standard_logits.shape == robust_logits.shape == (len(images), 1000)
    metrics = evaluator.metric_tensors(standard_logits, labels, reference)
    assert torch.all(metrics["exact_set_k5"] == 1)
    assert torch.all(metrics["exact_set_k2"] == 1)
    assert torch.isfinite(evaluator.boundary_margin(robust_logits, reference)).all()
    return standard, robust, images, labels, paths, reference


def test_strengthening_and_gate(
    standard: torch.nn.Module,
    robust: torch.nn.Module,
    images: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    with torch.no_grad():
        disagreement = standard(images).argmax(1).ne(robust(images).argmax(1))
    trajectories = evaluator.strengthen_trajectory(
        images,
        reference,
        robust,
        alpha_px=0.5,
        epsilon_px=4.0,
        capture_steps=(1, 2),
        top1_weight=1.0,
        set_weight=2.0,
        amp=False,
    )
    assert set(trajectories) == {1, 2}
    for value in trajectories.values():
        assert float((value - images).abs().max()) <= 4.0 / 255.0 + 1e-7
        assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0
    global_base, global_active = evaluator.gated_base(
        images, trajectories[2], disagreement, "global"
    )
    conditional_base, conditional_active = evaluator.gated_base(
        images, trajectories[2], disagreement, "disagreement"
    )
    assert global_active.all() and torch.equal(global_base, trajectories[2])
    assert torch.equal(conditional_active, disagreement)
    assert torch.equal(
        conditional_base,
        torch.where(disagreement[:, None, None, None], trajectories[2], images),
    )
    return {
        "candidate_id": "self_test_strength",
        "gate_mode": "global",
        "strength_alpha_px": 0.5,
        "strength_epsilon_px": 4.0,
        "strength_steps": 2,
        "top1_weight": 1.0,
        "set_weight": 2.0,
    }


def test_tiny_sigma_path(
    args: argparse.Namespace,
    standard: torch.nn.Module,
    robust: torch.nn.Module,
    probe_rows: list[dict[str, object]],
    selected: dict[str, object],
    device: torch.device,
) -> None:
    loader = evaluator.make_loader(
        evaluator.resolved(args.data_root), probe_rows[:2], 2, 0, device
    )
    tiny_args = argparse.Namespace(
        amp=False,
        sigma_forward_group=1,
        sigma_draws=2,
        mc_chunk=1,
        seed=12345,
    )
    original_grid = common.SIGMA_GRID
    try:
        common.SIGMA_GRID = (0.0, 0.005)
        rows = evaluator.evaluate_sigma_shard(
            "self_test", loader, standard, robust, selected, device, tiny_args
        )
    finally:
        common.SIGMA_GRID = original_grid
    assert len(rows) == 2 * 2 * 2
    stochastic = [row for row in rows if float(row["sigma"]) > 0]
    assert len(stochastic) == 4
    for row in stochastic:
        assert int(row["draws"]) == 2
        for key in common.PROBABILITY_METRICS.values():
            assert abs(float(row[key]) * 2 - round(float(row[key]) * 2)) < 1e-9


def main() -> None:
    args = parse_args()
    device_args = argparse.Namespace(
        device=args.device, gpu_memory_fraction=args.gpu_memory_fraction
    )
    device = evaluator.configure_device(device_args)
    evaluator.seed_everything(2026091101)
    test_frozen_grid()
    print("test_frozen_grid: OK")
    test_pure_selection()
    print("test_pure_selection: OK")
    test_bootstrap()
    print("test_bootstrap: OK")
    rows, shards = test_parent_and_partition(args)
    print("test_parent_and_partition: OK")
    standard, robust, images, _, _, reference = test_models_and_metrics(
        args, shards[0][:8], device
    )
    print("test_models_and_metrics: OK")
    selected = test_strengthening_and_gate(standard, robust, images, reference)
    print("test_strengthening_and_gate: OK")
    test_tiny_sigma_path(args, standard, robust, rows[:2], selected, device)
    print("test_tiny_sigma_path: OK")
    print("ImageNet defense calibration R2 self-test: 7/7 OK")


if __name__ == "__main__":
    main()
