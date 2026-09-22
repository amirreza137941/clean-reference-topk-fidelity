#!/usr/bin/env python3
"""Strict structural/statistical audit for CIFAR-100 defense calibration R1."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_cifar100_defense_calibration_common as common


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_calibration_train2k_r1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CIFAR-100 defense calibration R1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"Empty CSV: {path}")
    return rows


def assert_close(left: float, right: float, tolerance: float, message: str) -> None:
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > tolerance:
        raise AssertionError(f"{message}: {left} vs {right} (tol={tolerance})")


def verify_config(output_dir: Path) -> tuple[dict[str, Any], str]:
    config_path = output_dir / "run_config.json"
    config = common.read_json(config_path)
    if config.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Wrong calibration protocol_id")
    if config.get("parent_protocol_id") != common.PARENT_PROTOCOL_ID:
        raise AssertionError("Wrong R2 parent protocol_id")
    if config.get("offline") is not True:
        raise AssertionError("Calibration was not marked offline")
    for key in ("official_test_read", "attack_outcomes_read", "downloads_allowed"):
        if config.get(key) is not False:
            raise AssertionError(f"Data-governance violation: {key}={config.get(key)}")
    claimed = str(config.get("config_hash"))
    without_hash = dict(config)
    without_hash.pop("config_hash", None)
    actual = common.object_sha256(without_hash)
    if actual != claimed:
        raise AssertionError(f"run_config hash mismatch: {actual} != {claimed}")

    expected_checkpoint_hashes = {
        "standard": common.EXPECTED_STANDARD_SHA256,
        "anchor": common.EXPECTED_ANCHOR_SHA256,
    }
    for name, checkpoint in config["checkpoints"].items():
        path = Path(checkpoint["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen checkpoint is missing: {path}")
        if common.file_sha256(path) != checkpoint["sha256"]:
            raise AssertionError(f"Frozen checkpoint changed: {path}")
        if checkpoint["sha256"] != expected_checkpoint_hashes[name]:
            raise AssertionError(f"Unexpected frozen checkpoint hash for {name}")
    split_path = Path(config["split"]["path"])
    if not split_path.is_file() or common.file_sha256(split_path) != config["split"]["sha256"]:
        raise AssertionError("Frozen R2 split file changed")
    parent_manifest = config["parent_pipeline_manifest"]
    parent_manifest_path = Path(parent_manifest["path"])
    if (
        not parent_manifest_path.is_file()
        or common.file_sha256(parent_manifest_path) != parent_manifest["sha256"]
    ):
        raise AssertionError("Frozen R2 pipeline manifest changed")
    current_scripts = {
        "evaluator": PROJECT_DIR / "calibrate_topk_cifar100_actual_defense_r1.py",
        "calibration_common": PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
        "r2_common": PROJECT_DIR / "topk_cifar100_r2_common.py",
        "r2_anchor": PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    }
    if set(config["script_sha256"]) != set(current_scripts):
        raise AssertionError("Frozen script-hash key set is incomplete")
    for name, path in current_scripts.items():
        if not path.is_file() or common.file_sha256(path) != config["script_sha256"][name]:
            raise AssertionError(f"Frozen source changed or is missing: {path}")
    return config, claimed


def verify_subsets(output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = common.read_json(output_dir / "subset_manifest.json")
    if manifest.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Subset manifest protocol mismatch")
    if manifest.get("confirmation_used_for_retuning") is not False:
        raise AssertionError("Confirmation was marked as used for retuning")
    if manifest.get("validation_used_for_retuning") is not False:
        raise AssertionError("Validation was marked as used for retuning")
    if manifest.get("official_test_read") is not False or manifest.get("attack_outcomes_read") is not False:
        raise AssertionError("Subset manifest reports prohibited data use")

    tune = [int(value) for value in manifest["tune_indices"]]
    confirm = [int(value) for value in manifest["confirm_indices"]]
    validation = [int(value) for value in manifest["validation_indices"]]
    if len(tune) != int(config["tune_examples"]):
        raise AssertionError("Tune subset size mismatch")
    if len(confirm) != int(config["confirm_examples"]):
        raise AssertionError("Confirmation subset size mismatch")
    if len(validation) != int(config["validation_examples"]):
        raise AssertionError("Validation subset size mismatch")
    for name, values in (("tune", tune), ("confirm", confirm), ("validation", validation)):
        if len(values) != len(set(values)):
            raise AssertionError(f"Duplicate indices in {name}")
        if common.indices_sha256(values) != manifest[f"{name}_indices_sha256"]:
            raise AssertionError(f"{name} index hash mismatch")
    if set(tune) & set(confirm) or (set(tune) | set(confirm)) & set(validation):
        raise AssertionError("Calibration/confirmation/validation subsets overlap")

    parent = common.read_json(Path(config["split"]["path"]))
    parent_train = set(map(int, parent["train_indices"]))
    parent_validation = set(map(int, parent["validation_indices"]))
    if not set(tune).issubset(parent_train) or not set(confirm).issubset(parent_train):
        raise AssertionError("Calibration indices are not inside frozen R2 training")
    if not set(validation).issubset(parent_validation):
        raise AssertionError("Locked validation indices are not inside frozen R2 validation")
    if parent["train_indices_sha256"] != common.EXPECTED_TRAIN_HASH:
        raise AssertionError("Unexpected R2 training split hash")
    if parent["validation_indices_sha256"] != common.EXPECTED_VALIDATION_HASH:
        raise AssertionError("Unexpected R2 validation split hash")
    return manifest


def verify_selection(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    selection = common.read_json(output_dir / "calibration_selection.json")
    if selection.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Selection protocol mismatch")
    if selection.get("config_hash") != config["config_hash"]:
        raise AssertionError("Selection/config hash mismatch")
    claimed = selection.get("selection_hash")
    without_hash = dict(selection)
    without_hash.pop("selection_hash", None)
    if common.object_sha256(without_hash) != claimed:
        raise AssertionError("Selection hash mismatch")
    for key in ("confirmation_used_for_retuning", "validation_used_for_retuning", "official_test_read", "attack_outcomes_read"):
        if selection.get(key) is not False:
            raise AssertionError(f"Selection governance violation: {key}")

    strength_rows = read_csv(output_dir / "strength_calibration_summary.csv")
    expected_strength = 1 + len(config["strength_grid"])
    if len(strength_rows) != expected_strength:
        raise AssertionError(f"Expected {expected_strength} strength rows; got {len(strength_rows)}")
    robust = [row for row in strength_rows if row["method"] == "robust"]
    candidates = [row for row in strength_rows if row["method"] == "full_sigma0"]
    if len(robust) != 1 or len(candidates) != len(config["strength_grid"]):
        raise AssertionError("Malformed strength calibration rows")
    recomputed_strength, _ = common.select_strength_candidate(
        candidates, robust[0], float(config["max_clean_drop_pp"])
    )
    selected_strength = selection["strength"]
    if float(recomputed_strength["strength_alpha_px"]) != float(selected_strength["alpha_px"]):
        raise AssertionError("Selected alpha does not reproduce from frozen rule")
    if int(recomputed_strength["strength_steps"]) != int(selected_strength["steps"]):
        raise AssertionError("Selected step count does not reproduce from frozen rule")
    selected_flags = [row for row in candidates if row["selected"].lower() == "true"]
    if len(selected_flags) != 1:
        raise AssertionError("Strength CSV must mark exactly one candidate selected")

    sigma_rows = read_csv(output_dir / "sigma_calibration_summary.csv")
    if len(sigma_rows) != 2 * len(config["sigma_grid"]):
        raise AssertionError("Sigma calibration row count mismatch")
    pair_counts = Counter((round(float(row["sigma"]), 7), row["method"]) for row in sigma_rows)
    expected_pairs = {
        (round(float(sigma), 7), method)
        for sigma in config["sigma_grid"]
        for method in ("noise_only", "full")
    }
    if set(pair_counts) != expected_pairs or set(pair_counts.values()) != {1}:
        raise AssertionError("Sigma grid is incomplete or duplicated")
    recomputed_sigma, _ = common.select_sigma_candidate(
        sigma_rows, robust[0], recomputed_strength, float(config["max_clean_drop_pp"])
    )
    selected_sigma = float(selection["randomization"]["sigma"])
    if abs(float(recomputed_sigma["sigma"]) - selected_sigma) > 1e-12:
        raise AssertionError("Selected sigma does not reproduce from frozen rule")
    selected_sigma_rows = {
        row["method"] for row in sigma_rows if row["selected"].lower() == "true"
    }
    if selected_sigma_rows != {"noise_only", "full"}:
        raise AssertionError("Sigma CSV must select one complete method pair")
    sigma0_noise = next(
        row for row in sigma_rows
        if float(row["sigma"]) == 0.0 and row["method"] == "noise_only"
    )
    sigma0_full = next(
        row for row in sigma_rows
        if float(row["sigma"]) == 0.0 and row["method"] == "full"
    )
    for metric in common.PRIMARY_METRICS:
        assert_close(
            float(sigma0_noise[metric]), float(robust[0][metric]), 1e-10,
            f"sigma=0 Noise-only reproduction failed for {metric}",
        )
        assert_close(
            float(sigma0_full[metric]), float(recomputed_strength[metric]), 1e-10,
            f"sigma=0 Full reproduction failed for {metric}",
        )
    return selection, strength_rows, sigma_rows


def verify_partials(output_dir: Path, config: Mapping[str, Any], config_hash: str) -> None:
    expected_ids = {"tune_robust", "locked_confirmation", "locked_validation"}
    expected_ids |= {
        common.strength_candidate_id(candidate) for candidate in config["strength_grid"]
    }
    expected_ids |= {common.sigma_candidate_id(float(sigma)) for sigma in config["sigma_grid"]}
    paths = sorted((output_dir / "partials").glob("*.json"))
    if {path.stem for path in paths} != expected_ids:
        missing = sorted(expected_ids - {path.stem for path in paths})
        extra = sorted({path.stem for path in paths} - expected_ids)
        raise AssertionError(f"Partial set mismatch; missing={missing}, extra={extra}")
    for path in paths:
        payload = common.read_json(path)
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("config_hash") != config_hash:
            raise AssertionError(f"Partial config mismatch: {path}")
        if payload.get("condition_id") != path.stem:
            raise AssertionError(f"Partial condition/path mismatch: {path}")
        methods = [str(row["name"]) for row in payload["methods"]]
        counts = Counter(str(row["method"]) for row in payload["examples"])
        if set(methods) != set(counts) or len(counts) != len(methods):
            raise AssertionError(f"Partial method/example mismatch: {path}")


def verify_per_example(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    calibration = read_csv(output_dir / "calibration_per_example.csv")
    expected_calibration = int(config["tune_examples"]) * (
        1 + len(config["strength_grid"]) + 2 * len(config["sigma_grid"])
    )
    if len(calibration) != expected_calibration:
        raise AssertionError(
            f"Calibration row count {len(calibration)} != {expected_calibration}"
        )

    locked = read_csv(output_dir / "locked_clean_per_example.csv")
    expected_locked = 5 * (
        int(config["confirm_examples"]) + int(config["validation_examples"])
    )
    if len(locked) != expected_locked:
        raise AssertionError(f"Locked row count {len(locked)} != {expected_locked}")
    expected_methods = {"standard", "robust", "full_sigma0", "noise_only", "full"}
    for phase, count in (
        ("confirmation", int(config["confirm_examples"])),
        ("validation", int(config["validation_examples"])),
    ):
        phase_rows = [row for row in locked if row["phase"] == phase]
        if {row["method"] for row in phase_rows} != expected_methods:
            raise AssertionError(f"Locked {phase} method set mismatch")
        by_method = {
            method: sorted(
                (row for row in phase_rows if row["method"] == method),
                key=lambda row: int(row["sample_index"]),
            )
            for method in expected_methods
        }
        if any(len(rows) != count for rows in by_method.values()):
            raise AssertionError(f"Locked {phase} per-method counts mismatch")
        label_counts = Counter(int(row["label"]) for row in by_method["standard"])
        if set(label_counts) != set(range(100)) or set(label_counts.values()) != {count // 100}:
            raise AssertionError(f"Locked {phase} is not class-balanced")
        anchor = [(row["sample_index"], row["label"], row["reference_set"]) for row in by_method["standard"]]
        for method, rows in by_method.items():
            if [(row["sample_index"], row["label"], row["reference_set"]) for row in rows] != anchor:
                raise AssertionError(f"Locked {phase}/{method} samples or references differ")
        if any(abs(float(row["exact_set_probability"]) - 1.0) > 1e-12 for row in by_method["standard"]):
            raise AssertionError(f"Standard must match its own reference set in {phase}")

        for method, rows in by_method.items():
            for row in rows:
                draws = int(row["draws"])
                for key in (
                    "top1_accuracy_probability",
                    "reference_top1_match_probability",
                    "exact_set_probability",
                    "label_coverage_probability",
                ):
                    value = float(row[key])
                    if not -1e-12 <= value <= 1.0 + 1e-12:
                        raise AssertionError(f"Probability outside [0,1]: {phase}/{method}/{key}")
                    if abs(value * draws - round(value * draws)) > 1e-8:
                        raise AssertionError(f"Probability not representable by draw count: {phase}/{method}")
    return calibration, locked


def verify_locked_summaries(
    output_dir: Path,
    config: Mapping[str, Any],
    locked_rows: Sequence[Mapping[str, str]],
    final_summary: Mapping[str, Any],
) -> list[dict[str, str]]:
    summaries = read_csv(output_dir / "locked_clean_summary.csv")
    if len(summaries) != 10:
        raise AssertionError("Expected 5 methods x 2 locked phases")
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in locked_rows:
        grouped[(row["phase"], row["method"])].append(row)
    for summary in summaries:
        key = (summary["phase"], summary["method"])
        examples = grouped.get(key, [])
        if len(examples) != int(summary["examples"]):
            raise AssertionError(f"Locked summary count mismatch: {key}")
        for summary_key, example_key in (
            ("top1_accuracy_percent", "top1_accuracy_probability"),
            ("reference_top1_match_probability_percent", "reference_top1_match_probability"),
            ("exact_set_probability_percent", "exact_set_probability"),
            ("label_coverage_probability_percent", "label_coverage_probability"),
            ("mean_logits_top1_accuracy_percent", "mean_logits_top1_accuracy"),
            ("mean_logits_exact_set_percent", "mean_logits_exact_set"),
        ):
            observed = 100.0 * float(np.mean([float(row[example_key]) for row in examples]))
            assert_close(observed, float(summary[summary_key]), 1e-9, f"Summary mismatch {key}/{summary_key}")

    recomputed_gates = {
        phase: common.gate_locked_phase(
            [row for row in summaries if row["phase"] == phase],
            float(config["max_clean_drop_pp"]),
            float(config["locked_mc_tolerance_pp"]),
        )
        for phase in ("confirmation", "validation")
    }
    for phase, gate in recomputed_gates.items():
        saved = final_summary["locked_gates"][phase]
        if bool(saved["pass"]) != bool(gate["pass"]):
            raise AssertionError(f"Saved {phase} gate does not reproduce")
        for key, value in gate.items():
            if key != "pass":
                assert_close(float(saved[key]), float(value), 1e-10, f"Gate mismatch {phase}/{key}")
    selected_sigma = float(final_summary["selection"]["randomization"]["sigma"])
    expected_ready = bool(
        selected_sigma > 0
        and recomputed_gates["confirmation"]["pass"]
        and recomputed_gates["validation"]["pass"]
    )
    if bool(final_summary["attack_stage_ready"]) != expected_ready:
        raise AssertionError("attack_stage_ready does not reproduce")
    return summaries


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    config, config_hash = verify_config(output_dir)
    verify_subsets(output_dir, config)
    selection, strength_rows, sigma_rows = verify_selection(output_dir, config)
    verify_partials(output_dir, config, config_hash)
    _, locked_rows = verify_per_example(output_dir, config)
    final_summary = common.read_json(output_dir / "final_summary.json")
    if final_summary.get("status") != "PASS":
        raise AssertionError("Final summary status is not PASS")
    if final_summary.get("config_hash") != config_hash:
        raise AssertionError("Final summary/config mismatch")
    if final_summary.get("selection_hash") != selection["selection_hash"]:
        raise AssertionError("Final summary/selection mismatch")
    if final_summary.get("selection") != selection:
        raise AssertionError("Embedded final selection differs from the frozen selection file")
    summaries = verify_locked_summaries(output_dir, config, locked_rows, final_summary)

    bootstrap_path = output_dir / "validation_paired_bootstrap.csv"
    bootstrap = read_csv(bootstrap_path) if bootstrap_path.is_file() else []
    for row in bootstrap:
        left, right = row["left"], row["right"]
        metric_key = (
            "top1_accuracy_percent" if row["metric"] == "top1" else "exact_set_probability_percent"
        )
        table = {item["method"]: item for item in summaries if item["phase"] == "validation"}
        point = float(table[left][metric_key]) - float(table[right][metric_key])
        assert_close(point, float(row["difference_pp"]), 1e-9, f"Bootstrap point mismatch {row['comparison']}")

    print(f"PASS {output_dir}: frozen train-only calibration is structurally complete")
    print(
        "  selected strength: "
        f"alpha={float(selection['strength']['alpha_px']):.3f}/255, "
        f"epsilon={float(selection['strength']['epsilon_px']):.3f}/255, "
        f"steps={int(selection['strength']['steps'])}, "
        f"weights={float(selection['strength']['top1_weight']):g}:"
        f"{float(selection['strength']['set_weight']):g}"
    )
    print(f"  selected sigma: {float(selection['randomization']['sigma']):.7f}")
    for phase in ("confirmation", "validation"):
        print(f"Locked {phase}")
        table = {row["method"]: row for row in summaries if row["phase"] == phase}
        for method in ("standard", "robust", "full_sigma0", "noise_only", "full"):
            row = table[method]
            print(
                f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
                f"Exact={float(row['exact_set_probability_percent']):.3f}%"
            )
        gate = final_summary["locked_gates"][phase]
        print(f"  utility gate: {'PASS' if gate['pass'] else 'FAIL'}")
    if bootstrap:
        print("Validation paired comparisons")
        for row in bootstrap:
            if row["metric"] == "exact_set" or row["comparison"].startswith("full_minus_robust_top1"):
                print(
                    f"  {row['comparison']}: {float(row['difference_pp']):+.3f} pp "
                    f"[{float(row['ci95_low_pp']):+.3f}, {float(row['ci95_high_pp']):+.3f}]"
                )
    ready = bool(final_summary["attack_stage_ready"])
    print("ATTACK_STAGE_READY=" + ("YES" if ready else "NO"))
    if not ready:
        print("  blockers: " + "; ".join(final_summary["attack_stage_blockers"]))
    print("CIFAR-100 actual-defense calibration audit complete; no retraining, attack, or test-set read was performed.")


if __name__ == "__main__":
    main()
