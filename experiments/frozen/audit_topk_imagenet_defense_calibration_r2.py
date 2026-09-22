#!/usr/bin/env python3
"""Strict structural and statistical audit of ImageNet calibration R2."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import calibrate_topk_imagenet_defense_r2 as evaluator
import topk_imagenet_defense_calibration_r2_common as common
from topk_imagenet_preflight_common import (
    partition_rows,
    read_csv as read_r0_csv,
    validate_frozen_split,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = evaluator.DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ImageNet Top-k calibration R2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Not a boolean: {value!r}")


def assert_close(left: float, right: float, tolerance: float, label: str) -> None:
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > tolerance:
        raise AssertionError(f"{label}: {left} vs {right}, tolerance={tolerance}")


def grouped(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row[key])].append(row)
    return dict(output)


def verify_config(output_dir: Path) -> tuple[dict[str, Any], str]:
    config = common.read_json(output_dir / "run_config.json")
    if config.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Wrong R2 protocol")
    if config.get("parent_r0_protocol_id") != common.PARENT_R0_PROTOCOL_ID:
        raise AssertionError("Wrong R0 parent")
    if config.get("parent_r1_protocol_id") != common.PARENT_R1_PROTOCOL_ID:
        raise AssertionError("Wrong R1 parent")
    if int(config.get("primary_k")) != 5 or int(config.get("secondary_k")) != 2:
        raise AssertionError("Frozen k values changed")
    if int(config.get("calibration_examples")) != common.CALIBRATION_EXAMPLES:
        raise AssertionError("Calibration sample count changed")
    if int(config.get("calibration_shards")) != common.CALIBRATION_SHARDS:
        raise AssertionError("Calibration shard count changed")
    if int(config.get("sigma_draws")) != 50:
        raise AssertionError("Calibration MC count changed")
    if abs(float(config.get("max_clean_drop_pp")) - 2.0) > 1e-12:
        raise AssertionError("Non-inferiority margin changed")
    if tuple(config.get("gate_modes", [])) != common.GATE_MODES:
        raise AssertionError("Gate grid changed")
    if config.get("strength_grid") != list(common.STRENGTH_GRID):
        raise AssertionError("Strength grid changed")
    if tuple(float(value) for value in config.get("sigma_grid", [])) != common.SIGMA_GRID:
        raise AssertionError("Sigma grid changed")
    if config.get("partitions_read") != ["calibration_dev"]:
        raise AssertionError("R2 read an unauthorized partition")
    if set(config.get("forbidden_partitions_not_read", [])) != {
        "attack_pilot", "attack_confirmatory", "clean_confirmation"
    }:
        raise AssertionError("Forbidden-partition declaration changed")
    for key, expected in (
        ("offline", True),
        ("downloads_allowed", False),
        ("training_allowed", False),
        ("attack_outcomes_read", False),
        ("common_random_numbers", True),
    ):
        if config.get(key) is not expected:
            raise AssertionError(f"Governance/config violation: {key}={config.get(key)}")
    claimed_hash = str(config["config_hash"])
    hash_payload = dict(config)
    hash_payload.pop("config_hash", None)
    if common.object_sha256(hash_payload) != claimed_hash:
        raise AssertionError("run_config hash mismatch")

    expected_sources = {
        filename: PROJECT_DIR / filename
        for filename in common.NEW_SCRIPT_NAMES
    }
    expected_sources.update(
        {filename: PROJECT_DIR / filename for filename in common.EXPECTED_PARENT_SOURCE_HASHES}
    )
    if set(config["script_sha256"]) != set(expected_sources):
        raise AssertionError("Recorded source set changed")
    for name, path in expected_sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if common.file_sha256(path) != config["script_sha256"][name]:
            raise AssertionError(f"Source changed after run: {name}")
    for name, expected in common.EXPECTED_PARENT_SOURCE_HASHES.items():
        if config["script_sha256"][name] != expected:
            raise AssertionError(f"Unexpected parent source hash: {name}")

    for name, expected in (
        ("standard", common.STANDARD_CHECKPOINT_SHA256),
        ("robust", common.ROBUST_CHECKPOINT_SHA256),
    ):
        item = config["checkpoints"][name]
        if item["sha256"] != expected:
            raise AssertionError(f"Recorded checkpoint SHA mismatch: {name}")
        if common.file_sha256(Path(item["path"])) != expected:
            raise AssertionError(f"Live checkpoint changed: {name}")
    if config["parent_artifacts"] != {
        "r1_handoff_sha256": common.R1_HANDOFF_SHA256,
        "r1_report_sha256": common.R1_REPORT_SHA256,
        "frozen_split_sha256": common.FROZEN_SPLIT_SHA256,
    }:
        raise AssertionError("Parent artifact locks changed")
    return config, claimed_hash


def verify_manifest(output_dir: Path, config: Mapping[str, Any]) -> list[dict[str, str]]:
    manifest = common.read_json(output_dir / "calibration_dev_manifest.json")
    rows = common.read_csv(output_dir / "calibration_dev_manifest.csv")
    if manifest.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Calibration manifest protocol mismatch")
    if manifest.get("partition") != "calibration_dev":
        raise AssertionError("Wrong calibration partition")
    if int(manifest.get("examples")) != 2000 or int(manifest.get("classes")) != 1000:
        raise AssertionError("Calibration manifest dimensions changed")
    if len(rows) != 2000 or len({row["relative_path"] for row in rows}) != 2000:
        raise AssertionError("Calibration manifest rows are incomplete or duplicated")
    counts = Counter(int(row["target"]) for row in rows)
    if set(counts) != set(range(1000)) or set(counts.values()) != {2}:
        raise AssertionError("Calibration manifest is not two-per-class")
    shard_counts = Counter(row["shard_id"] for row in rows)
    expected_shards = {common.shard_id(index) for index in range(common.CALIBRATION_SHARDS)}
    if set(shard_counts) != expected_shards or set(shard_counts.values()) != {100}:
        raise AssertionError("Calibration shard membership changed")
    canonical = []
    for row in rows:
        canonical.append(
            {
                "partition": row["partition"],
                "target": int(row["target"]),
                "class_name": row["class_name"],
                "within_class_slot": int(row["within_class_slot"]),
                "relative_path": row["relative_path"],
                "shard_id": row["shard_id"],
            }
        )
    if common.object_sha256(canonical) != manifest["rows_sha256"]:
        raise AssertionError("Calibration manifest row hash mismatch")

    r0_dir = Path(config["paths"]["r0_dir"])
    split_rows = read_r0_csv(r0_dir / "frozen_imagenet_val_split.csv")
    if validate_frozen_split(split_rows)["split_sha256"] != common.FROZEN_SPLIT_SHA256:
        raise AssertionError("Live R0 split changed")
    calibration = partition_rows(split_rows, "calibration_dev")
    shards = evaluator.outcome_blind_shards(calibration)
    expected = []
    for shard_number, shard_rows in enumerate(shards):
        for row in shard_rows:
            expected.append({**row, "shard_id": common.shard_id(shard_number)})
    if canonical != expected:
        raise AssertionError("Calibration manifest does not reproduce from R0")
    return rows


def verify_probability_row(row: Mapping[str, Any], expected_draws: int) -> None:
    if int(row["draws"]) != expected_draws:
        raise AssertionError("Per-example draw count mismatch")
    for key in common.PROBABILITY_METRICS.values():
        value = float(row[key])
        if not -1e-12 <= value <= 1.0 + 1e-12:
            raise AssertionError(f"Probability outside [0,1]: {key}={value}")
        if abs(value * expected_draws - round(value * expected_draws)) > 1e-8:
            raise AssertionError(f"Probability is not representable by {expected_draws} draws")


def verify_partials(
    output_dir: Path,
    config_hash: str,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    membership = grouped(manifest_rows, "shard_id")
    strength_selection = common.read_json(output_dir / "strength_selection.json")
    selection_hash = str(strength_selection["strength_selection_hash"])
    all_rows: dict[str, list[dict[str, Any]]] = {"strength": [], "sigma": []}
    expected_candidates = {
        "standard",
        "identity",
        *(
            common.strength_candidate_id(gate, candidate)
            for candidate in common.STRENGTH_GRID
            for gate in common.GATE_MODES
        ),
    }
    expected_sigma_candidates = {
        f"sigma_{common.sigma_tag(sigma)}_{method}"
        for sigma in common.SIGMA_GRID
        for method in ("noise_only", "full")
    }
    for phase, candidates, selection in (
        ("strength", expected_candidates, None),
        ("sigma", expected_sigma_candidates, selection_hash),
    ):
        paths = sorted((output_dir / "partials" / phase).glob("shard_*.json"))
        if len(paths) != common.CALIBRATION_SHARDS:
            raise AssertionError(f"{phase} partial count mismatch")
        for path in paths:
            payload = common.read_json(path)
            shard = path.stem
            if payload.get("protocol_id") != common.PROTOCOL_ID:
                raise AssertionError(f"Partial protocol mismatch: {path}")
            if payload.get("config_hash") != config_hash:
                raise AssertionError(f"Partial config mismatch: {path}")
            if payload.get("phase") != phase or payload.get("shard_id") != shard:
                raise AssertionError(f"Partial phase/shard mismatch: {path}")
            if payload.get("selection_hash") != selection:
                raise AssertionError(f"Partial selection mismatch: {path}")
            rows = [dict(row) for row in payload["examples"]]
            per_candidate = grouped(rows, "candidate_id")
            if set(per_candidate) != candidates:
                raise AssertionError(f"Candidate set mismatch: {path}")
            expected_paths = sorted(row["relative_path"] for row in membership[shard])
            for candidate_id, candidate_rows in per_candidate.items():
                if len(candidate_rows) != 100:
                    raise AssertionError(f"Candidate row count mismatch: {path}/{candidate_id}")
                if sorted(row["relative_path"] for row in candidate_rows) != expected_paths:
                    raise AssertionError(f"Candidate membership mismatch: {path}/{candidate_id}")
                draws = int(candidate_rows[0]["draws"])
                if phase == "strength" or float(candidate_rows[0]["sigma"]) == 0.0:
                    if draws != 1:
                        raise AssertionError("Deterministic condition must use one draw")
                else:
                    if draws != 50:
                        raise AssertionError("Stochastic condition must use 50 draws")
                for row in candidate_rows:
                    verify_probability_row(row, draws)
                    if candidate_id.startswith("global_") and int(row["gate_active"]) != 1:
                        raise AssertionError("Global strengthening gate was not active")
                    if candidate_id.endswith("noise_only") and int(row["gate_active"]) != 0:
                        raise AssertionError("Noise-only activated strengthening")
            reference_anchor = {
                row["relative_path"]: (
                    row["label"], row["reference_top1"],
                    row["reference_set_k5"], row["reference_set_k2"]
                )
                for row in next(iter(per_candidate.values()))
            }
            for candidate_rows in per_candidate.values():
                observed = {
                    row["relative_path"]: (
                        row["label"], row["reference_top1"],
                        row["reference_set_k5"], row["reference_set_k2"]
                    )
                    for row in candidate_rows
                }
                if observed != reference_anchor:
                    raise AssertionError(f"References differ across candidates: {path}")
            all_rows[phase].extend(rows)
    return all_rows["strength"], all_rows["sigma"]


SUMMARY_FIELDS = (
    "gate_activation_percent",
    "top1_accuracy_percent",
    "reference_top1_match_probability_percent",
    "label_coverage_k5_probability_percent",
    "exact_set_k5_probability_percent",
    "label_coverage_k2_probability_percent",
    "exact_set_k2_probability_percent",
    "mean_logits_top1_accuracy_percent",
    "mean_logits_label_coverage_k5_percent",
    "mean_logits_exact_set_k5_percent",
    "mean_logits_exact_set_k2_percent",
    "mean_boundary_margin_k5",
    "mean_boundary_margin_k2",
)


def compare_summaries(
    saved: Sequence[Mapping[str, Any]],
    recomputed: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    saved_map = {str(row["candidate_id"]): row for row in saved}
    recomputed_map = {str(row["candidate_id"]): row for row in recomputed}
    if set(saved_map) != set(recomputed_map):
        raise AssertionError(f"{label} summary candidate mismatch")
    for candidate in saved_map:
        for field in SUMMARY_FIELDS:
            assert_close(
                float(saved_map[candidate][field]),
                float(recomputed_map[candidate][field]),
                1e-9,
                f"{label}/{candidate}/{field}",
            )


def compare_bootstrap(
    saved: Sequence[Mapping[str, Any]],
    recomputed: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    key = lambda row: (str(row["group_id"]), str(row["metric"]), str(row["comparison"]))
    saved_map = {key(row): row for row in saved}
    recomputed_map = {key(row): row for row in recomputed}
    if set(saved_map) != set(recomputed_map):
        raise AssertionError(f"{label} bootstrap key mismatch")
    for item in saved_map:
        for field in ("difference_pp", "ci95_low_pp", "ci95_high_pp"):
            assert_close(
                float(saved_map[item][field]), float(recomputed_map[item][field]),
                1e-10, f"{label}/{item}/{field}"
            )
        if str(saved_map[item]["status"]) != str(recomputed_map[item]["status"]):
            raise AssertionError(f"{label} status mismatch: {item}")


def verify_strength_frontier(
    output_dir: Path,
    config: Mapping[str, Any],
    strength_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    saved_rows = common.read_csv(output_dir / "strength_frontier_per_example.csv")
    if len(saved_rows) != 56 * 2000:
        raise AssertionError("Strength per-example shape mismatch")
    if {
        (row["candidate_id"], row["relative_path"]) for row in saved_rows
    } != {
        (str(row["candidate_id"]), str(row["relative_path"])) for row in strength_rows
    }:
        raise AssertionError("Strength aggregate CSV differs from partials")
    recomputed_summaries = evaluator.summarize_rows(saved_rows)
    saved_summaries = common.read_csv(output_dir / "strength_frontier_summary.csv")
    compare_summaries(saved_summaries, recomputed_summaries, "strength")
    robust_rows = evaluator.rows_for_candidate(saved_rows, "identity")
    candidate_summaries = [row for row in recomputed_summaries if row["candidate_id"] != "standard"]
    specs = []
    for summary in candidate_summaries:
        candidate_id = str(summary["candidate_id"])
        candidate_rows = evaluator.rows_for_candidate(saved_rows, candidate_id)
        for metric in ("top1", "top5", "exact_k5"):
            specs.append(
                evaluator.bootstrap_spec(
                    candidate_id, candidate_id, "robust", metric,
                    candidate_rows, robust_rows
                )
            )
    bootstrap = common.paired_bootstrap_many(
        specs,
        int(config["bootstrap_resamples"]),
        int(config["bootstrap_seeds"]["strength"]),
    )
    saved_bootstrap = common.read_csv(output_dir / "strength_paired_bootstrap.csv")
    compare_bootstrap(saved_bootstrap, bootstrap, "strength")
    selected, _ = common.select_strength_candidate(
        candidate_summaries, bootstrap, float(config["max_clean_drop_pp"])
    )
    selected_flags = [
        row for row in saved_summaries
        if row["candidate_id"] != "standard" and as_bool(row["selected"])
    ]
    if len(selected_flags) != 1 or selected_flags[0]["candidate_id"] != selected["candidate_id"]:
        raise AssertionError("Strength selected flag does not reproduce")
    strength_selection = common.read_json(output_dir / "strength_selection.json")
    hash_payload = dict(strength_selection)
    claimed_hash = hash_payload.pop("strength_selection_hash")
    if common.object_sha256(hash_payload) != claimed_hash:
        raise AssertionError("Strength selection hash mismatch")
    if strength_selection["candidate_id"] != selected["candidate_id"]:
        raise AssertionError("Strength selection JSON does not reproduce")
    return strength_selection, saved_rows, robust_rows


def verify_sigma_frontier(
    output_dir: Path,
    config: Mapping[str, Any],
    sigma_rows: Sequence[Mapping[str, Any]],
    strength_rows: Sequence[Mapping[str, Any]],
    robust_rows: Sequence[Mapping[str, Any]],
    strength_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    saved_rows = common.read_csv(output_dir / "sigma_frontier_per_example.csv")
    if len(saved_rows) != 26 * 2000:
        raise AssertionError("Sigma per-example shape mismatch")
    if {
        (row["candidate_id"], row["relative_path"]) for row in saved_rows
    } != {
        (str(row["candidate_id"]), str(row["relative_path"])) for row in sigma_rows
    }:
        raise AssertionError("Sigma aggregate CSV differs from partials")
    recomputed_summaries = evaluator.summarize_rows(saved_rows)
    saved_summaries = common.read_csv(output_dir / "sigma_frontier_summary.csv")
    compare_summaries(saved_summaries, recomputed_summaries, "sigma")
    selected_strength_rows = evaluator.rows_for_candidate(
        strength_rows, str(strength_selection["candidate_id"])
    )
    noise0 = evaluator.rows_for_candidate(saved_rows, "sigma_0_noise_only")
    full0 = evaluator.rows_for_candidate(saved_rows, "sigma_0_full")
    for left, right, label in (
        (noise0, robust_rows, "noise0/robust"),
        (full0, selected_strength_rows, "full0/strength"),
    ):
        for metric_key in common.PROBABILITY_METRICS.values():
            drift = float(np.max(np.abs(common.aligned_delta(left, right, metric_key))))
            if drift > 1e-10:
                raise AssertionError(f"{label} mismatch for {metric_key}: {drift}")

    summary_map = {str(row["candidate_id"]): row for row in recomputed_summaries}
    specs = []
    combined = []
    for sigma in common.SIGMA_GRID:
        tag = common.sigma_tag(sigma)
        group_id = f"sigma_{tag}"
        noise_id = f"sigma_{tag}_noise_only"
        full_id = f"sigma_{tag}_full"
        noise = evaluator.rows_for_candidate(saved_rows, noise_id)
        full = evaluator.rows_for_candidate(saved_rows, full_id)
        definitions = (
            ("noise_top1_vs_robust", "noise_only", "robust", noise, robust_rows, "top1"),
            ("noise_top5_vs_robust", "noise_only", "robust", noise, robust_rows, "top5"),
            ("noise_exact_k5_vs_robust", "noise_only", "robust", noise, robust_rows, "exact_k5"),
            ("full_top1_vs_robust", "full", "robust", full, robust_rows, "top1"),
            ("full_top5_vs_robust", "full", "robust", full, robust_rows, "top5"),
            ("full_top1_vs_full_sigma0", "full", "full_sigma0", full, full0, "top1"),
            ("full_top5_vs_full_sigma0", "full", "full_sigma0", full, full0, "top5"),
            ("full_exact_k5_vs_full_sigma0", "full", "full_sigma0", full, full0, "exact_k5"),
        )
        for constraint, left_name, right_name, left, right, base_metric in definitions:
            spec = evaluator.bootstrap_spec(
                group_id, left_name, right_name, constraint, left, right,
                common.PROBABILITY_METRICS[base_metric]
            )
            spec["base_metric"] = base_metric
            specs.append(spec)
        noise_summary = summary_map[noise_id]
        full_summary = summary_map[full_id]
        combined.append(
            {
                "sigma": float(sigma),
                "noise_top1_accuracy_percent": noise_summary["top1_accuracy_percent"],
                "noise_top5_accuracy_percent": noise_summary["label_coverage_k5_probability_percent"],
                "noise_exact_set_k5_probability_percent": noise_summary["exact_set_k5_probability_percent"],
                "full_top1_accuracy_percent": full_summary["top1_accuracy_percent"],
                "full_top5_accuracy_percent": full_summary["label_coverage_k5_probability_percent"],
                "full_exact_set_k5_probability_percent": full_summary["exact_set_k5_probability_percent"],
            }
        )
    bootstrap = common.paired_bootstrap_many(
        specs,
        int(config["bootstrap_resamples"]),
        int(config["bootstrap_seeds"]["sigma"]),
    )
    saved_bootstrap = common.read_csv(output_dir / "sigma_paired_bootstrap.csv")
    compare_bootstrap(saved_bootstrap, bootstrap, "sigma")
    selected, _ = common.select_sigma_candidate(
        combined, bootstrap, float(config["max_clean_drop_pp"])
    )
    selected_flags = {
        (round(float(row["sigma"]), 7), row["method"])
        for row in saved_summaries if as_bool(row["selected"])
    }
    expected_flags = {
        (round(float(selected["sigma"]), 7), "noise_only"),
        (round(float(selected["sigma"]), 7), "full"),
    }
    if selected_flags != expected_flags:
        raise AssertionError("Sigma selected flags do not reproduce")
    return selected, saved_rows


def verify_selected(
    output_dir: Path,
    config: Mapping[str, Any],
    strength_selection: Mapping[str, Any],
    sigma_selection: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    rows = common.read_csv(output_dir / "selected_dev_per_example.csv")
    summaries = common.read_csv(output_dir / "selected_dev_summary.csv")
    if len(rows) != 5 * 2000 or len(summaries) != 5:
        raise AssertionError("Selected output shape mismatch")
    expected_methods = {"standard", "robust", "full_sigma0", "noise_only", "full"}
    if {row["method"] for row in summaries} != expected_methods:
        raise AssertionError("Selected method set mismatch")
    recomputed = evaluator.summarize_rows(rows)
    compare_summaries(summaries, recomputed, "selected")
    groups = grouped(rows, "method")
    anchors = {
        method: sorted(
            (
                row["relative_path"], row["label"], row["reference_top1"],
                row["reference_set_k5"], row["reference_set_k2"]
            )
            for row in method_rows
        )
        for method, method_rows in groups.items()
    }
    if len({tuple(value) for value in anchors.values()}) != 1:
        raise AssertionError("Selected methods do not share samples/labels/references")
    specs = []
    for left, right, metrics in (
        ("noise_only", "robust", ("top1", "top5", "exact_k5", "exact_k2")),
        ("full_sigma0", "robust", ("top1", "top5", "exact_k5", "exact_k2")),
        ("full", "robust", ("top1", "top5", "exact_k5", "exact_k2")),
        ("full", "noise_only", ("top1", "top5", "exact_k5", "exact_k2")),
    ):
        for metric in metrics:
            specs.append(
                evaluator.bootstrap_spec(
                    "selected", left, right, metric, groups[left], groups[right]
                )
            )
    bootstrap = common.paired_bootstrap_many(
        specs,
        int(config["bootstrap_resamples"]),
        int(config["bootstrap_seeds"]["selected"]),
    )
    saved_bootstrap = common.read_csv(output_dir / "selected_dev_paired_bootstrap.csv")
    compare_bootstrap(saved_bootstrap, bootstrap, "selected")

    decomposition = common.read_csv(output_dir / "reference_quality_decomposition.csv")
    recomputed_decomposition = evaluator.reference_quality_decomposition(rows)
    key = lambda row: (str(row["group"]), str(row["method"]))
    saved_map = {key(row): row for row in decomposition}
    recomputed_map = {key(row): row for row in recomputed_decomposition}
    if set(saved_map) != set(recomputed_map) or len(saved_map) != 15:
        raise AssertionError("Reference-quality decomposition shape mismatch")
    for item in saved_map:
        for field in (
            "top1_accuracy_percent",
            "label_coverage_k5_probability_percent",
            "exact_set_k5_probability_percent",
            "exact_set_k2_probability_percent",
        ):
            assert_close(
                float(saved_map[item][field]), float(recomputed_map[item][field]),
                1e-9, f"decomposition/{item}/{field}"
            )

    table = {row["method"]: row for row in summaries}
    if table["full_sigma0"]["source_candidate_id"] != strength_selection["candidate_id"]:
        raise AssertionError("Selected strength source mismatch")
    sigma = round(float(sigma_selection["sigma"]), 7)
    if round(float(table["full"]["sigma"]), 7) != sigma:
        raise AssertionError("Selected Full sigma mismatch")
    if round(float(table["noise_only"]["sigma"]), 7) != sigma:
        raise AssertionError("Selected Noise-only sigma mismatch")
    return table, saved_bootstrap


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    config, config_hash = verify_config(output_dir)
    manifest_rows = verify_manifest(output_dir, config)
    strength_partial_rows, sigma_partial_rows = verify_partials(
        output_dir, config_hash, manifest_rows
    )
    strength_selection, strength_rows, robust_rows = verify_strength_frontier(
        output_dir, config, strength_partial_rows
    )
    sigma_selection, _ = verify_sigma_frontier(
        output_dir, config, sigma_partial_rows, strength_rows, robust_rows,
        strength_selection
    )
    table, selected_bootstrap = verify_selected(
        output_dir, config, strength_selection, sigma_selection
    )

    selection = common.read_json(output_dir / "defense_selection_r2.json")
    hash_payload = dict(selection)
    claimed_selection_hash = hash_payload.pop("selection_hash")
    if common.object_sha256(hash_payload) != claimed_selection_hash:
        raise AssertionError("Final defense selection hash mismatch")
    if selection["config_hash"] != config_hash:
        raise AssertionError("Defense selection config mismatch")
    if selection["strength"]["candidate_id"] != strength_selection["candidate_id"]:
        raise AssertionError("Defense/strength selection mismatch")
    if abs(float(selection["randomization"]["sigma"]) - float(sigma_selection["sigma"])) > 1e-12:
        raise AssertionError("Defense/sigma selection mismatch")

    final = common.read_json(output_dir / "final_summary.json")
    if final.get("protocol_id") != common.PROTOCOL_ID or final.get("status") != "PASS":
        raise AssertionError("Final summary status/protocol mismatch")
    if final.get("config_hash") != config_hash:
        raise AssertionError("Final summary config mismatch")
    if final.get("selection_hash") != claimed_selection_hash or final.get("selection") != selection:
        raise AssertionError("Final summary selection mismatch")
    if final.get("attack_stage_ready") is not False:
        raise AssertionError("Attack stage was incorrectly authorized")
    if final.get("paper_claim_authorized") is not False:
        raise AssertionError("Calibration incorrectly authorized a paper claim")
    expected_ready = bool(
        selection["strength"]["positive_strength_selected"]
        and selection["randomization"]["positive_sigma_selected"]
        and selection["strength"]["selected_dev_summary"]["jointly_ci_feasible"]
        and selection["randomization"]["selected_dev_constraints"]["jointly_ci_feasible"]
    )
    if bool(final["official_clean_stage_ready"]) != expected_ready:
        raise AssertionError("Official-clean readiness does not reproduce")

    comparison_map = {row["comparison"]: row for row in selected_bootstrap}
    print(
        f"PASS {output_dir}: 2,000 calibration_dev images, 40 resumable partials, "
        "frontiers and paired CIs reproduce"
    )
    print(
        "  selected strength: "
        f"{selection['strength']['candidate_id']}, gate={selection['strength']['gate_mode']}, "
        f"alpha={float(selection['strength']['alpha_px']):.3f}/255, "
        f"epsilon={float(selection['strength']['epsilon_px']):.3f}/255, "
        f"steps={int(selection['strength']['steps'])}, "
        f"weights=1:{float(selection['strength']['set_weight']):g}"
    )
    print(
        f"  selected sigma={float(selection['randomization']['sigma']):.7f}, "
        f"calibration MC={int(selection['randomization']['draws_during_calibration'])}"
    )
    print("Selected calibration_dev operating point")
    for method in ("standard", "robust", "full_sigma0", "noise_only", "full"):
        row = table[method]
        print(
            f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Top5={float(row['label_coverage_k5_probability_percent']):.3f}% "
            f"Exact5={float(row['exact_set_k5_probability_percent']):.3f}% "
            f"Exact2={float(row['exact_set_k2_probability_percent']):.3f}%"
        )
    for name in (
        "full_minus_robust_top1",
        "full_minus_robust_top5",
        "full_minus_robust_exact_k5",
        "full_minus_noise_only_exact_k5",
    ):
        row = comparison_map[name]
        print(
            f"  {name}: {float(row['difference_pp']):+.3f} pp "
            f"[{float(row['ci95_low_pp']):+.3f}, {float(row['ci95_high_pp']):+.3f}]"
        )
    print("OFFICIAL_CLEAN_STAGE_READY=" + ("YES" if expected_ready else "NO"))
    if not expected_ready:
        print("  blockers: " + "; ".join(final["official_clean_stage_blockers"]))
    print("ATTACK_STAGE_READY=NO")
    print(
        "ImageNet R2 audit complete; no training, attack, download, or outcome read "
        "outside calibration_dev was performed."
    )


if __name__ == "__main__":
    main()
