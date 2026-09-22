#!/usr/bin/env python3
"""Strict dependency-light audit for CIFAR-100 official clean confirmation R3."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_cifar100_official_clean_r3_common as common


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_official_clean_confirmation_k2_mc100_r3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CIFAR-100 official clean R3")
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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Not a boolean: {value!r}")


def assert_close(left: float, right: float, tolerance: float, message: str) -> None:
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > tolerance:
        raise AssertionError(f"{message}: {left} vs {right} (tol={tolerance})")


def grouped(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row[key])].append(row)
    return dict(output)


def verify_config(output_dir: Path) -> tuple[dict[str, Any], str]:
    config = common.read_json(output_dir / "run_config.json")
    if config.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Wrong official-clean protocol")
    if config.get("parent_dev_protocol_id") != common.PARENT_DEV_PROTOCOL_ID:
        raise AssertionError("Wrong parent dev protocol")
    if config.get("parent_model_protocol_id") != common.PARENT_MODEL_PROTOCOL_ID:
        raise AssertionError("Wrong parent model protocol")
    for key in ("offline", "no_training", "no_hyperparameter_search"):
        if config.get(key) is not True:
            raise AssertionError(f"Frozen-governance flag missing: {key}")
    for key in ("downloads_allowed", "official_test_used_for_defense_selection", "attack_outcomes_read"):
        if config.get(key) is not False:
            raise AssertionError(f"Data-governance violation: {key}")
    expected_scalars = {
        "test_examples": 10000,
        "shard_count": 20,
        "shard_size": 500,
        "mc_draws": 100,
        "seed": 2026090501,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 2026090502,
        "attack_split_seed": 2026090503,
        "batch_size": 25,
        "mc_chunk": 20,
        "primary_k": 2,
    }
    for key, expected in expected_scalars.items():
        if int(config.get(key)) != expected:
            raise AssertionError(f"Frozen config changed: {key}={config.get(key)}")
    if config.get("amp") is not False:
        raise AssertionError("Primary R3 evaluation must use full precision")
    if float(config.get("noninferiority_margin_pp")) != 2.0:
        raise AssertionError("Non-inferiority margin changed")
    if config.get("parent_dev_config_hash") != common.EXPECTED_DEV_CONFIG_HASH:
        raise AssertionError("Parent dev config hash changed")
    if config.get("parent_dev_selection_hash") != common.EXPECTED_DEV_SELECTION_HASH:
        raise AssertionError("Parent dev selection hash changed")
    if config.get("selection") != common.EXPECTED_SELECTION:
        raise AssertionError("Frozen defense operating point changed")

    claimed_hash = str(config.get("config_hash"))
    payload = dict(config)
    payload.pop("config_hash", None)
    if common.object_sha256(payload) != claimed_hash:
        raise AssertionError("run_config self-hash mismatch")

    expected_sources = {
        "official_evaluator": PROJECT_DIR / "evaluate_topk_cifar100_official_clean_r3.py",
        "official_common": PROJECT_DIR / "topk_cifar100_official_clean_r3_common.py",
        "dev_evaluator": PROJECT_DIR / "calibrate_topk_cifar100_dev_pareto_r2.py",
        "dev_common": PROJECT_DIR / "topk_cifar100_dev_pareto_r2_common.py",
        "r1_evaluator": PROJECT_DIR / "calibrate_topk_cifar100_actual_defense_r1.py",
        "r1_common": PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
        "r2_common": PROJECT_DIR / "topk_cifar100_r2_common.py",
        "r2_anchor": PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    }
    if set(config["script_sha256"]) != set(expected_sources):
        raise AssertionError("Source-hash key set changed")
    for name, path in expected_sources.items():
        if not path.is_file() or common.file_sha256(path) != config["script_sha256"][name]:
            raise AssertionError(f"Frozen source changed or is missing: {name}: {path}")

    checkpoints = config["checkpoints"]
    expected_weights = {
        "standard": common.EXPECTED_STANDARD_SHA256,
        "anchor": common.EXPECTED_ANCHOR_SHA256,
    }
    for name, expected in expected_weights.items():
        item = checkpoints[name]
        if item["sha256"] != expected or common.file_sha256(Path(item["path"])) != expected:
            raise AssertionError(f"Checkpoint mismatch: {name}")
    for name, item in config["parent_artifacts"].items():
        path = Path(item["path"])
        if not path.is_file() or common.file_sha256(path) != item["sha256"]:
            raise AssertionError(f"Parent artifact changed: {name}: {path}")
    return config, claimed_hash


def verify_manifest(output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = output_dir / "official_test_manifest.json"
    manifest = common.read_json(path)
    if manifest.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Official-test manifest protocol mismatch")
    claimed = str(manifest.get("manifest_hash"))
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    if common.object_sha256(payload) != claimed:
        raise AssertionError("Official-test manifest self-hash mismatch")
    config_item = config["official_evaluation_manifest"]
    if config_item["manifest_hash"] != claimed or common.file_sha256(path) != config_item["sha256"]:
        raise AssertionError("Official-test manifest/config mismatch")
    indices = [int(value) for value in manifest["test_indices"]]
    if indices != list(range(10000)):
        raise AssertionError("Evaluation is not the complete ordered official test set")
    if common.sequence_sha256(indices) != manifest["test_indices_sha256"]:
        raise AssertionError("Official-test index hash mismatch")
    if [int(value) for value in manifest["class_counts"]] != [100] * 100:
        raise AssertionError("Official-test class counts changed")
    if manifest.get("official_test_used_for_defense_parameter_selection") is not False:
        raise AssertionError("Official test was marked as used for defense selection")
    if manifest.get("attack_outcomes_read") is not False:
        raise AssertionError("Attack outcomes were read before the clean gate")
    if manifest.get("defense_parameters_frozen_before_test_read") is not True:
        raise AssertionError("Defense freeze declaration is absent")
    return manifest


def verify_partials(
    output_dir: Path, config: Mapping[str, Any], config_hash: str
) -> list[dict[str, Any]]:
    paths = sorted((output_dir / "partials").glob("official_test_shard_*.json"))
    if [path.stem for path in paths] != [f"official_test_shard_{i:02d}" for i in range(20)]:
        raise AssertionError("Expected exactly 20 official-test shard partials")
    all_rows: list[dict[str, Any]] = []
    method_specs = {str(row["name"]): row for row in config["methods"]}
    if tuple(method_specs) != common.METHODS:
        raise AssertionError("Frozen method order changed")
    for shard_id, path in enumerate(paths):
        payload = common.read_json(path)
        expected_indices = list(range(shard_id * 500, (shard_id + 1) * 500))
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("config_hash") != config_hash:
            raise AssertionError(f"Partial config mismatch: {path}")
        if payload.get("condition_id") != path.stem or int(payload.get("shard_id")) != shard_id:
            raise AssertionError(f"Partial identity mismatch: {path}")
        if [int(value) for value in payload["sample_indices"]] != expected_indices:
            raise AssertionError(f"Partial index range mismatch: {path}")
        if common.sequence_sha256(expected_indices) != payload["sample_indices_sha256"]:
            raise AssertionError(f"Partial index hash mismatch: {path}")
        if payload["methods"] != config["methods"]:
            raise AssertionError(f"Partial method specs changed: {path}")
        summaries = payload["summaries"]
        rows = payload["examples"]
        if Counter(str(row["method"]) for row in summaries) != Counter(common.METHODS):
            raise AssertionError(f"Partial summary methods changed: {path}")
        for method in common.METHODS:
            selected = [row for row in rows if str(row["method"]) == method]
            if [int(row["sample_index"]) for row in selected] != expected_indices:
                raise AssertionError(f"Partial sample rows changed: {path}/{method}")
            spec = method_specs[method]
            expected_draws = int(spec["draws"] if float(spec["sigma"]) > 0 else 1)
            for row in selected:
                if row.get("protocol_id") != common.PROTOCOL_ID:
                    raise AssertionError(f"Row protocol mismatch: {path}/{method}")
                if row.get("config_hash") != config_hash or int(row["shard_id"]) != shard_id:
                    raise AssertionError(f"Row identity mismatch: {path}/{method}")
                if int(row["draws"]) != expected_draws:
                    raise AssertionError(f"Draw count mismatch: {path}/{method}")
                for metric_key in common.PROBABILITY_METRICS.values():
                    value = float(row[metric_key])
                    if not -1e-12 <= value <= 1.0 + 1e-12:
                        raise AssertionError(f"Probability outside [0,1]: {path}/{method}")
                    if abs(value * expected_draws - round(value * expected_draws)) > 1e-8:
                        raise AssertionError(f"Probability cannot be represented by draws: {path}/{method}")
                expected_gate = 1 if method in {"full_sigma0", "full"} else 0
                if int(row["gate_active"]) != expected_gate:
                    raise AssertionError(f"Global gate mismatch: {path}/{method}")
        all_rows.extend(dict(row) for row in rows)
    return all_rows


def verify_alignment(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_method = grouped(rows, "method")
    if set(by_method) != set(common.METHODS):
        raise AssertionError("Aggregate method set mismatch")
    maps = {method: common.rows_by_index(by_method[method]) for method in common.METHODS}
    if any(sorted(mapping) != list(range(10000)) for mapping in maps.values()):
        raise AssertionError("Aggregate methods do not cover the full official test")
    for index in range(10000):
        anchors = {
            (
                int(maps[method][index]["label"]),
                int(maps[method][index]["reference_top1"]),
                str(maps[method][index]["reference_set"]),
            )
            for method in common.METHODS
        }
        if len(anchors) != 1:
            raise AssertionError(f"Labels/references differ across methods at sample {index}")
        if float(maps["standard"][index]["reference_top1_match_probability"]) != 1.0:
            raise AssertionError(f"Standard/reference Top-1 mismatch at sample {index}")
        if float(maps["standard"][index]["exact_set_probability"]) != 1.0:
            raise AssertionError(f"Standard/reference set mismatch at sample {index}")
    return by_method


def verify_aggregate_csv(
    output_dir: Path,
    partial_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, list[Mapping[str, Any]]]]:
    rows = read_csv(output_dir / "official_clean_per_example.csv")
    if len(rows) != 50000:
        raise AssertionError(f"Expected 50,000 aggregate rows; got {len(rows)}")
    canonical_partial = sorted(
        (
            str(row["method"]),
            int(row["sample_index"]),
            int(row["label"]),
            float(row["top1_accuracy_probability"]),
            float(row["exact_set_probability"]),
        )
        for row in partial_rows
    )
    canonical_csv = sorted(
        (
            str(row["method"]),
            int(row["sample_index"]),
            int(row["label"]),
            float(row["top1_accuracy_probability"]),
            float(row["exact_set_probability"]),
        )
        for row in rows
    )
    if canonical_csv != canonical_partial:
        raise AssertionError("Aggregate per-example CSV differs from partials")
    return rows, verify_alignment(rows)


def verify_summaries(
    output_dir: Path, by_method: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Mapping[str, Any]]:
    summaries = read_csv(output_dir / "official_clean_summary.csv")
    if Counter(row["method"] for row in summaries) != Counter(common.METHODS):
        raise AssertionError("Summary method set mismatch")
    summary_map = {str(row["method"]): row for row in summaries}
    for method in common.METHODS:
        rows = by_method[method]
        summary = summary_map[method]
        if int(summary["examples"]) != 10000:
            raise AssertionError(f"Summary example count mismatch: {method}")
        for summary_key, row_key in (
            ("top1_accuracy_percent", "top1_accuracy_probability"),
            ("reference_top1_match_probability_percent", "reference_top1_match_probability"),
            ("exact_set_probability_percent", "exact_set_probability"),
            ("label_coverage_probability_percent", "label_coverage_probability"),
            ("gate_activation_percent", "gate_active"),
        ):
            observed = 100.0 * float(np.mean([float(row[row_key]) for row in rows]))
            assert_close(observed, float(summary[summary_key]), 1e-9, f"Summary mismatch {method}/{summary_key}")
    assert_close(float(summary_map["standard"]["top1_accuracy_percent"]), 67.78, 1e-9, "Standard parent anchor")
    assert_close(float(summary_map["robust"]["top1_accuracy_percent"]), 65.41, 1e-9, "Robust parent Top-1")
    assert_close(float(summary_map["robust"]["exact_set_probability_percent"]), 46.21, 1e-9, "Robust parent Exact-set")
    return summary_map


def verify_comparisons(
    output_dir: Path, by_method: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, str]]:
    comparisons = read_csv(output_dir / "official_clean_paired_bootstrap.csv")
    expected_names = {
        "standard_minus_robust_top1",
        "noise_only_minus_robust_top1",
        "noise_only_minus_robust_exact_set",
        "full_sigma0_minus_robust_top1",
        "full_sigma0_minus_robust_exact_set",
        "full_minus_robust_top1",
        "full_minus_robust_exact_set",
        "full_minus_noise_only_top1",
        "full_minus_noise_only_exact_set",
        "full_minus_full_sigma0_top1",
        "full_minus_full_sigma0_exact_set",
    }
    if {row["comparison"] for row in comparisons} != expected_names:
        raise AssertionError("Paired comparison set mismatch")
    for row in comparisons:
        left = row["left"]
        right = row["right"]
        metric = row["metric"]
        delta = common.aligned_delta(
            by_method[left], by_method[right], common.PROBABILITY_METRICS[metric]
        )
        assert_close(
            100.0 * float(delta.mean()),
            float(row["difference_pp"]),
            1e-9,
            f"Paired point mismatch: {row['comparison']}",
        )
        low = float(row["ci95_low_pp"])
        point = float(row["difference_pp"])
        high = float(row["ci95_high_pp"])
        if not low <= point <= high:
            raise AssertionError(f"Malformed CI: {row['comparison']}")
        if int(row["examples"]) != 10000 or int(row["resamples"]) != 10000:
            raise AssertionError(f"Paired protocol count mismatch: {row['comparison']}")
        if int(row["seed"]) != 2026090502:
            raise AssertionError(f"Bootstrap seed mismatch: {row['comparison']}")
    return comparisons


def verify_secondary_tables(
    output_dir: Path, by_method: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    decomposition = read_csv(output_dir / "official_clean_reference_decomposition.csv")
    if len(decomposition) != 15:
        raise AssertionError("Reference decomposition must contain 3 groups x 5 methods")
    expected_groups = {"all", "standard_correct", "standard_wrong"}
    if {row["group"] for row in decomposition} != expected_groups:
        raise AssertionError("Reference decomposition groups changed")
    classes = read_csv(output_dir / "official_clean_classwise.csv")
    if len(classes) != 500:
        raise AssertionError("Classwise table must contain 100 classes x 5 methods")
    if {int(row["class_id"]) for row in classes} != set(range(100)):
        raise AssertionError("Classwise table does not cover all classes")
    for method in common.METHODS:
        if sum(row["method"] == method for row in classes) != 100:
            raise AssertionError(f"Classwise method count mismatch: {method}")


def verify_attack_partitions(manifest: Mapping[str, Any], standard_rows: Sequence[Mapping[str, Any]]) -> None:
    label_map = {int(row["sample_index"]): int(row["label"]) for row in standard_rows}
    ordered_labels = [label_map[index] for index in range(10000)]
    if common.sequence_sha256(ordered_labels) != manifest["labels_sha256"]:
        raise AssertionError("Official-test label hash differs from evaluated rows")
    if np.bincount(np.asarray(ordered_labels), minlength=100).astype(int).tolist() != [100] * 100:
        raise AssertionError("Evaluated official-test labels are not class-balanced")
    partitions = manifest["future_attack_partitions"]
    pilot = [int(value) for value in partitions["pilot"]["indices"]]
    confirmatory = [int(value) for value in partitions["confirmatory"]["indices"]]
    if len(pilot) != 200 or len(set(pilot)) != 200:
        raise AssertionError("Pilot attack partition is not 200 unique examples")
    if len(confirmatory) != 1000 or len(set(confirmatory)) != 1000:
        raise AssertionError("Confirmatory attack partition is not 1,000 unique examples")
    if set(pilot) & set(confirmatory):
        raise AssertionError("Future attack partitions overlap")
    for name, indices, per_class in (("pilot", pilot, 2), ("confirmatory", confirmatory, 10)):
        if common.sequence_sha256(indices) != partitions[name]["indices_sha256"]:
            raise AssertionError(f"{name} attack partition hash mismatch")
        counts = Counter(label_map[index] for index in indices)
        if counts != Counter({class_id: per_class for class_id in range(100)}):
            raise AssertionError(f"{name} attack partition is not class-balanced")


def verify_decisions_and_final(
    output_dir: Path,
    config_hash: str,
    comparisons: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    expected_decisions, expected_ready = common.evaluate_gate_decisions(comparisons, 2.0)
    saved = read_csv(output_dir / "official_clean_gate_decisions.csv")
    if len(saved) != len(expected_decisions):
        raise AssertionError("Gate-decision count mismatch")
    saved_map = {row["gate"]: row for row in saved}
    for expected in expected_decisions:
        row = saved_map.get(str(expected["gate"]))
        if row is None:
            raise AssertionError(f"Missing gate decision: {expected['gate']}")
        if as_bool(row["passed"]) != bool(expected["passed"]):
            raise AssertionError(f"Gate decision mismatch: {expected['gate']}")
        for key in ("difference_pp", "ci95_low_pp", "ci95_high_pp", "threshold_pp"):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"Gate value mismatch {expected['gate']}/{key}")

    handoff = common.read_json(output_dir / "attack_handoff.json")
    claimed_handoff_hash = str(handoff.get("handoff_hash"))
    payload = dict(handoff)
    payload.pop("handoff_hash", None)
    if common.object_sha256(payload) != claimed_handoff_hash:
        raise AssertionError("Attack handoff self-hash mismatch")
    if handoff.get("protocol_id") != common.PROTOCOL_ID or handoff.get("config_hash") != config_hash:
        raise AssertionError("Attack handoff identity mismatch")
    if bool(handoff.get("authorized")) != expected_ready:
        raise AssertionError("Attack authorization mismatch")
    if handoff.get("frozen_defense") != common.EXPECTED_SELECTION:
        raise AssertionError("Attack handoff changed the frozen defense")

    final = common.read_json(output_dir / "final_summary.json")
    if final.get("protocol_id") != common.PROTOCOL_ID or final.get("config_hash") != config_hash:
        raise AssertionError("Final summary identity mismatch")
    if final.get("status") != "PASS":
        raise AssertionError("Evaluation did not complete structurally")
    if bool(final.get("attack_stage_ready")) != expected_ready:
        raise AssertionError("Final attack-stage decision mismatch")
    expected_label = "PASS" if expected_ready else "FAIL"
    if final.get("official_clean_confirmation") != expected_label:
        raise AssertionError("Final clean-confirmation label mismatch")
    reproduction = final.get("parent_test_reproduction", {})
    if reproduction.get("passed") is not True:
        raise AssertionError("Parent official-test deterministic anchors did not reproduce")
    return expected_decisions, expected_ready


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    config, config_hash = verify_config(output_dir)
    manifest = verify_manifest(output_dir, config)
    partial_rows = verify_partials(output_dir, config, config_hash)
    _, by_method = verify_aggregate_csv(output_dir, partial_rows)
    summaries = verify_summaries(output_dir, by_method)
    comparisons = verify_comparisons(output_dir, by_method)
    verify_secondary_tables(output_dir, by_method)
    verify_attack_partitions(manifest, by_method["standard"])
    decisions, attack_ready = verify_decisions_and_final(
        output_dir, config_hash, comparisons
    )

    print(f"PASS {output_dir}: 20 shards, 50,000 rows, 500 classwise cells", flush=True)
    for method in common.METHODS:
        row = summaries[method]
        print(
            f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Exact={float(row['exact_set_probability_percent']):.3f}%",
            flush=True,
        )
    print("PASS deterministic parent anchors: standard Top1=67.780%, robust Top1=65.410%, robust Exact=46.210%", flush=True)
    print("Preregistered official clean gates", flush=True)
    for row in decisions:
        print(
            f"  {row['gate']}: {float(row['difference_pp']):+.3f} pp "
            f"[{float(row['ci95_low_pp']):+.3f}, {float(row['ci95_high_pp']):+.3f}] "
            f"-> {'PASS' if row['passed'] else 'FAIL'}",
            flush=True,
        )
    print("OFFICIAL_CLEAN_CONFIRMATION=" + ("PASS" if attack_ready else "FAIL"), flush=True)
    print("ATTACK_STAGE_READY=" + ("YES" if attack_ready else "NO"), flush=True)
    print("Official-clean R3 audit complete; no training, retuning, attack, merge, or deletion was performed.", flush=True)


if __name__ == "__main__":
    main()
