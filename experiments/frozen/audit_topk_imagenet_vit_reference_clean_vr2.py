#!/usr/bin/env python3
"""Independent auditor for ImageNet ViT-reference clean confirmation VR2."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import confirm_topk_imagenet_vit_reference_clean_vr2 as evaluator
import topk_imagenet_vit_reference_clean_vr2_common as common
from topk_imagenet_preflight_common import (
    partition_rows,
    read_csv as read_r0_csv,
    validate_frozen_split,
)


DEFAULT_OUTPUT_DIR = evaluator.DEFAULT_OUTPUT_DIR
PROBABILITY_KEYS = tuple(common.PROBABILITY_METRICS.values())
MEAN_BINARY_KEYS = (
    "mean_logits_top1_accuracy",
    "mean_logits_label_coverage_k5",
    "mean_logits_exact_set_k5",
    "mean_logits_exact_set_k2",
)
NUMERIC_ROW_KEYS = {
    "label",
    "reference_top1",
    "gate_active",
    "draws",
    "sigma",
    "strength_alpha_px",
    "strength_epsilon_px",
    "strength_steps",
    "top1_weight",
    "set_weight",
    *PROBABILITY_KEYS,
    *MEAN_BINARY_KEYS,
    "mean_boundary_margin_k5",
    "mean_boundary_margin_k2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit official ImageNet ViT-reference clean confirmation VR2"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def assert_close(left: float, right: float, tolerance: float, label: str) -> None:
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        raise AssertionError(f"Non-finite value for {label}: {left}, {right}")
    if abs(float(left) - float(right)) > float(tolerance):
        raise AssertionError(f"{label} differs: {left} vs {right}")


def compare_table(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
    tolerance: float = 1e-10,
) -> None:
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], Mapping[str, Any]]:
        output: dict[tuple[str, ...], Mapping[str, Any]] = {}
        for row in rows:
            key = tuple(str(row[field]) for field in key_fields)
            if key in output:
                raise AssertionError(f"Duplicate table key: {key}")
            output[key] = row
        return output

    left = index(observed)
    right = index(expected)
    if set(left) != set(right):
        raise AssertionError(
            f"Table keys differ: missing={set(right)-set(left)}, extra={set(left)-set(right)}"
        )
    for key in sorted(left):
        a = left[key]
        b = right[key]
        if set(a) != set(b):
            raise AssertionError(f"Schema differs for {key}: {set(a) ^ set(b)}")
        for field in a:
            try:
                av = float(a[field])
                bv = float(b[field])
            except (TypeError, ValueError):
                if str(a[field]) != str(b[field]):
                    raise AssertionError(f"{key}/{field} differs: {a[field]} vs {b[field]}")
            else:
                assert_close(av, bv, tolerance, f"{key}/{field}")


def verify_config(output_dir: Path) -> tuple[dict[str, Any], str, str]:
    protocol = common.read_json(output_dir / "frozen_protocol_vr2.json")
    expected_protocol = common.frozen_protocol_payload()
    if protocol != expected_protocol:
        raise AssertionError("Frozen VR2 preregistration changed")
    protocol_hash = str(protocol["protocol_hash"])
    config_path = output_dir / "run_config.json"
    config = common.read_json(config_path)
    hash_payload = dict(config)
    claimed = str(hash_payload.pop("config_hash"))
    if common.object_sha256(hash_payload) != claimed:
        raise AssertionError("run_config hash does not reproduce")
    if config.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Unexpected clean-confirmation protocol")
    if config.get("protocol_hash") != protocol_hash:
        raise AssertionError("Config preregistration hash mismatch")
    if config.get("parent_protocol_id") != common.PARENT_PROTOCOL_ID:
        raise AssertionError("Unexpected calibration parent protocol")
    if config.get("stage") != "official_clean_confirmation":
        raise AssertionError("Unexpected stage in run_config")
    if int(config.get("clean_examples", -1)) != common.CLEAN_EXAMPLES:
        raise AssertionError("Official clean sample count changed")
    if int(config.get("clean_shards", -1)) != common.CLEAN_SHARDS:
        raise AssertionError("Official clean shard count changed")
    if int(config.get("mc_draws", -1)) != common.MC_DRAWS:
        raise AssertionError("Official clean MC draw count changed")
    bootstrap = config.get("bootstrap", {})
    if bootstrap.get("unit") != "ImageNet class cluster":
        raise AssertionError("Bootstrap sampling unit changed")
    if int(bootstrap.get("clusters", -1)) != common.CLEAN_CLASSES:
        raise AssertionError("Bootstrap cluster count changed")
    if int(bootstrap.get("resamples", -1)) != common.BOOTSTRAP_RESAMPLES:
        raise AssertionError("Bootstrap resample count changed")
    if int(bootstrap.get("seed", -1)) != common.BOOTSTRAP_SEED:
        raise AssertionError("Bootstrap seed changed")
    if int(config.get("seed", -1)) != common.EVALUATION_SEED:
        raise AssertionError("Evaluation seed changed")
    if int(config.get("batch_size", -1)) != 16 or int(config.get("mc_chunk", -1)) != 4:
        raise AssertionError("Frozen batching configuration changed")
    if bool(config.get("amp")):
        raise AssertionError("AMP must be disabled")
    if config.get("selection_allowed") is not False:
        raise AssertionError("Selection must be disabled")
    if config.get("attack_allowed") is not False:
        raise AssertionError("Attack must be disabled")
    if config.get("attack_outcomes_read") is not False:
        raise AssertionError("Attack outcomes were read")
    if config.get("image_partitions_read") != [common.CLEAN_PARTITION]:
        raise AssertionError("Unexpected image partition provenance")
    parent = config.get("parent_artifacts", {})
    if parent.get("selection_hash") != common.EXPECTED_SELECTION_HASH:
        raise AssertionError("Config selection hash mismatch")
    if parent.get("calibration_config_hash") != common.EXPECTED_CALIBRATION_CONFIG_HASH:
        raise AssertionError("Config calibration hash mismatch")
    if config.get("architecture_roles") != {
        "standard_reference": "torchvision ViT-B/16",
        "defended_backbone": "torchvision ResNet-50",
        "cross_architecture_reference": True,
    }:
        raise AssertionError("Cross-architecture role declaration changed")
    checkpoints = config.get("checkpoints", {})
    if checkpoints.get("standard", {}).get("sha256") != common.STANDARD_CHECKPOINT_SHA256:
        raise AssertionError("Standard checkpoint declaration changed")
    if checkpoints.get("robust", {}).get("sha256") != common.ROBUST_CHECKPOINT_SHA256:
        raise AssertionError("Robust checkpoint declaration changed")
    if (
        checkpoints.get("robust", {}).get("state_dict_fingerprint_sha256")
        != common.ROBUST_STATE_FINGERPRINT
    ):
        raise AssertionError("Robust state declaration changed")

    recorded = config.get("script_sha256", {})
    project_dir = Path(__file__).resolve().parent
    expected_names = (
        set(common.NEW_SCRIPT_NAMES)
        | set(common.EXPECTED_CALIBRATION_SOURCE_HASHES)
        | set(evaluator.calibration.common.EXPECTED_PARENT_SOURCE_HASHES)
    )
    if set(recorded) != expected_names:
        raise AssertionError("Recorded source-file set differs from the frozen protocol")
    for filename, recorded_hash in recorded.items():
        source = project_dir / filename
        if not source.is_file() or common.file_sha256(source) != recorded_hash:
            raise AssertionError(f"Source hash mismatch: {filename}")
    for filename, expected in common.EXPECTED_CALIBRATION_SOURCE_HASHES.items():
        if recorded.get(filename) != expected:
            raise AssertionError(f"Frozen calibration source mismatch: {filename}")
    return config, claimed, protocol_hash


def verify_manifest(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    manifest = common.read_json(output_dir / "clean_confirmation_manifest.json")
    rows = common.read_csv(output_dir / "clean_confirmation_manifest.csv")
    if manifest.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Manifest protocol mismatch")
    if manifest.get("protocol_hash") != config.get("protocol_hash"):
        raise AssertionError("Manifest preregistration mismatch")
    if manifest.get("partition") != common.CLEAN_PARTITION:
        raise AssertionError("Manifest partition mismatch")
    if int(manifest.get("examples", -1)) != common.CLEAN_EXAMPLES:
        raise AssertionError("Manifest example count mismatch")
    if int(manifest.get("examples_per_class", -1)) != common.CLEAN_EXAMPLES_PER_CLASS:
        raise AssertionError("Manifest class balance mismatch")
    if int(manifest.get("shards", -1)) != common.CLEAN_SHARDS:
        raise AssertionError("Manifest shard count mismatch")
    if manifest.get("selection_hash") != common.EXPECTED_SELECTION_HASH:
        raise AssertionError("Manifest selection hash mismatch")
    if manifest.get("parent_split_sha256") != common.FROZEN_SPLIT_SHA256:
        raise AssertionError("Manifest parent split mismatch")
    if manifest.get("image_partitions_read") != [common.CLEAN_PARTITION]:
        raise AssertionError("Manifest records an unexpected image partition")
    typed_rows = [
        {
            "partition": str(row["partition"]),
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "within_class_slot": int(row["within_class_slot"]),
            "relative_path": str(row["relative_path"]),
            "shard_id": str(row["shard_id"]),
        }
        for row in rows
    ]
    if len(typed_rows) != common.CLEAN_EXAMPLES:
        raise AssertionError("Manifest CSV row count mismatch")
    if common.object_sha256(typed_rows) != manifest.get("rows_sha256"):
        raise AssertionError("Manifest row hash mismatch")
    if len({row["relative_path"] for row in typed_rows}) != common.CLEAN_EXAMPLES:
        raise AssertionError("Manifest contains duplicate paths")
    target_counts = Counter(int(row["target"]) for row in typed_rows)
    if set(target_counts) != set(range(1000)) or set(target_counts.values()) != {
        common.CLEAN_EXAMPLES_PER_CLASS
    }:
        raise AssertionError("Manifest is not 45-per-class balanced")
    shard_paths: dict[str, list[str]] = defaultdict(list)
    for row in typed_rows:
        shard_paths[str(row["shard_id"])].append(str(row["relative_path"]))
    expected_shards = {common.shard_id(index) for index in range(common.CLEAN_SHARDS)}
    if set(shard_paths) != expected_shards:
        raise AssertionError("Manifest shard IDs differ")
    if any(len(paths) != common.EXAMPLES_PER_SHARD for paths in shard_paths.values()):
        raise AssertionError("Manifest shards are not all size 500")

    r0_dir = Path(config["paths"]["r0_dir"])
    split_rows = read_r0_csv(r0_dir / "frozen_imagenet_val_split.csv")
    split_check = validate_frozen_split(split_rows)
    if split_check.get("split_sha256") != common.FROZEN_SPLIT_SHA256:
        raise AssertionError("Parent frozen split hash changed")
    clean_rows = partition_rows(split_rows, common.CLEAN_PARTITION)
    clean_by_path = {str(row["relative_path"]): row for row in clean_rows}
    if set(clean_by_path) != {str(row["relative_path"]) for row in typed_rows}:
        raise AssertionError("Manifest membership differs from clean_confirmation")
    for row in typed_rows:
        parent = clean_by_path[str(row["relative_path"])]
        for field in ("partition", "target", "class_name", "within_class_slot"):
            if str(row[field]) != str(parent[field]):
                raise AssertionError(f"Manifest parent mismatch for {row['relative_path']}/{field}")
    return rows, shard_paths


def verify_model_audit(output_dir: Path, config: Mapping[str, Any]) -> None:
    current = common.read_json(output_dir / "model_load_audit.json")
    parent_path = Path(config["paths"]["calibration_dir"]) / "model_load_audit.json"
    parent = common.read_json(parent_path)
    if current != parent:
        raise AssertionError("Model-load audit differs from the audited VR1 parent")
    if (
        current.get("robust", {}).get("state_dict_fingerprint_sha256")
        != common.ROBUST_STATE_FINGERPRINT
    ):
        raise AssertionError("Robust state fingerprint mismatch")
    standard_arch = current.get("standard", {}).get("architecture", {})
    if standard_arch.get("architecture_id") != "torchvision_vit_b_16":
        raise AssertionError("Standard reference is not torchvision ViT-B/16")
    robust_arch = current.get("robust", {}).get("architecture", {})
    if robust_arch.get("architecture_id") != "torchvision_resnet50":
        raise AssertionError("Defended backbone is not torchvision ResNet-50")


def verify_probability(value: Any, draws: int, label: str) -> None:
    number = float(value)
    if not -1e-12 <= number <= 1.0 + 1e-12:
        raise AssertionError(f"Probability outside [0,1] for {label}: {number}")
    scaled = number * int(draws)
    if abs(scaled - round(scaled)) > 1e-8:
        raise AssertionError(f"Probability is not a {draws}-draw proportion for {label}")


def verify_method_parameters(row: Mapping[str, Any]) -> None:
    method = str(row["method"])
    if str(row["candidate_id"]) != method:
        raise AssertionError(f"Candidate alias mismatch for {method}")
    draws = int(row["draws"])
    if method in common.STOCHASTIC_METHODS:
        if draws != common.MC_DRAWS or abs(float(row["sigma"]) - 0.025) > 1e-12:
            raise AssertionError(f"Stochastic settings changed for {method}")
    else:
        if draws != 1 or abs(float(row["sigma"])) > 1e-12:
            raise AssertionError(f"Deterministic settings changed for {method}")
    if method in {"full_sigma0", "full"}:
        expected = {
            "gate_mode": "global",
            "gate_active": 1,
            "strength_alpha_px": 1.0,
            "strength_epsilon_px": 4.0,
            "strength_steps": 4,
            "top1_weight": 1.0,
            "set_weight": 4.0,
        }
    else:
        expected = {
            "gate_mode": "none",
            "gate_active": 0,
            "strength_alpha_px": 0.0,
            "strength_epsilon_px": 0.0,
            "strength_steps": 0,
            "top1_weight": 0.0,
            "set_weight": 0.0,
        }
    for key, value in expected.items():
        if isinstance(value, str):
            if str(row[key]) != value:
                raise AssertionError(f"{method}/{key} changed")
        elif abs(float(row[key]) - float(value)) > 1e-12:
            raise AssertionError(f"{method}/{key} changed")
    for key in PROBABILITY_KEYS:
        verify_probability(row[key], draws, f"{method}/{key}")
    for key in MEAN_BINARY_KEYS:
        verify_probability(row[key], 1, f"{method}/{key}")
    if method == "standard":
        for key in (
            "reference_top1_match_probability",
            "exact_set_k5_probability",
            "exact_set_k2_probability",
            "mean_logits_exact_set_k5",
            "mean_logits_exact_set_k2",
        ):
            assert_close(float(row[key]), 1.0, 1e-12, f"standard/{key}")


def verify_partials(
    output_dir: Path,
    config_hash: str,
    protocol_hash: str,
    shard_paths: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    partial_dir = output_dir / "partials" / "official_clean"
    files = sorted(partial_dir.glob("*.json"))
    expected_names = {f"{common.shard_id(index)}.json" for index in range(common.CLEAN_SHARDS)}
    if {path.name for path in files} != expected_names:
        raise AssertionError("Partial-file set differs from the 90 frozen shards")
    all_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for path in files:
        payload = common.read_json(path)
        shard = path.stem
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("config_hash") != config_hash:
            raise AssertionError(f"Partial config mismatch: {path}")
        if payload.get("protocol_hash") != protocol_hash:
            raise AssertionError(f"Partial preregistration mismatch: {path}")
        if payload.get("selection_hash") != common.EXPECTED_SELECTION_HASH:
            raise AssertionError(f"Partial selection mismatch: {path}")
        if (
            payload.get("phase") != "official_clean"
            or payload.get("partition") != common.CLEAN_PARTITION
            or payload.get("shard_id") != shard
        ):
            raise AssertionError(f"Partial phase/shard mismatch: {path}")
        if int(payload.get("input_examples", -1)) != common.EXAMPLES_PER_SHARD:
            raise AssertionError(f"Partial input count mismatch: {path}")
        rows = payload.get("examples", [])
        if len(rows) != common.EXAMPLES_PER_SHARD * len(common.METHODS):
            raise AssertionError(f"Partial row count mismatch: {path}")
        by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("protocol_id") != common.PROTOCOL_ID:
                raise AssertionError(f"Row protocol mismatch: {path}")
            if (
                row.get("phase") != "official_clean"
                or row.get("partition") != common.CLEAN_PARTITION
                or row.get("shard_id") != shard
            ):
                raise AssertionError(f"Row phase/shard mismatch: {path}")
            verify_method_parameters(row)
            by_path[str(row["relative_path"])].append(row)
        if sorted(by_path) != sorted(shard_paths[shard]):
            raise AssertionError(f"Partial membership mismatch: {path}")
        for relative_path, group in by_path.items():
            if {str(row["method"]) for row in group} != set(common.METHODS):
                raise AssertionError(f"Method coverage mismatch for {relative_path}")
            labels = {int(row["label"]) for row in group}
            refs1 = {int(row["reference_top1"]) for row in group}
            refs5 = {str(row["reference_set_k5"]) for row in group}
            refs2 = {str(row["reference_set_k2"]) for row in group}
            if any(len(values) != 1 for values in (labels, refs1, refs5, refs2)):
                raise AssertionError(f"Cross-method target/reference mismatch for {relative_path}")
            if len(next(iter(refs5)).split("|")) != common.PRIMARY_K:
                raise AssertionError(f"Malformed Top-5 reference for {relative_path}")
            if len(next(iter(refs2)).split("|")) != common.SECONDARY_K:
                raise AssertionError(f"Malformed Top-2 reference for {relative_path}")
        all_rows.extend(rows)
        runtime_rows.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "shard_id": shard,
                "input_examples": int(payload["input_examples"]),
                "output_rows": len(rows),
                "wall_seconds": float(payload["wall_seconds"]),
            }
        )
    if len(all_rows) != common.CLEAN_EXAMPLES * len(common.METHODS):
        raise AssertionError("Merged per-example row count mismatch")
    return all_rows, runtime_rows


def verify_per_example_csv(
    output_dir: Path, expected_rows: Iterable[Mapping[str, Any]]
) -> tuple[Path, int]:
    path = output_dir / "official_clean_per_example.csv.gz"
    reader = iter(common.iter_csv(path))
    count = 0
    for expected in expected_rows:
        try:
            observed = next(reader)
        except StopIteration as exc:
            raise AssertionError("Compressed per-example CSV ended early") from exc
        count += 1
        if set(expected) != set(observed):
            raise AssertionError(f"Per-example CSV schema mismatch at row {count}")
        for key in expected:
            if key in NUMERIC_ROW_KEYS:
                assert_close(
                    float(expected[key]),
                    float(observed[key]),
                    1e-12,
                    f"per-example row {count}/{key}",
                )
            elif str(expected[key]) != str(observed[key]):
                raise AssertionError(f"Per-example CSV mismatch at row {count}/{key}")
    try:
        extra = next(reader)
    except StopIteration:
        extra = None
    if extra is not None:
        raise AssertionError("Compressed per-example CSV contains extra rows")
    expected_count = common.CLEAN_EXAMPLES * len(common.METHODS)
    if count != expected_count:
        raise AssertionError(f"Per-example CSV has {count} rows, expected {expected_count}")
    return path, count


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    config, config_hash, protocol_hash = verify_config(output_dir)
    _manifest_rows, shard_paths = verify_manifest(output_dir, config)
    verify_model_audit(output_dir, config)
    rows, runtime_rows = verify_partials(
        output_dir, config_hash, protocol_hash, shard_paths
    )
    per_example_path, per_example_count = verify_per_example_csv(
        output_dir, iter(rows)
    )

    expected_summaries = evaluator.summarize_rows(rows)
    saved_summaries = common.read_csv(output_dir / "official_clean_summary.csv")
    compare_table(saved_summaries, expected_summaries, ("method",))

    expected_runtime = runtime_rows
    saved_runtime = common.read_csv(output_dir / "partial_runtime_summary.csv")
    compare_table(saved_runtime, expected_runtime, ("shard_id",))

    print("Recomputing 50,000-resample paired class-cluster bootstrap", flush=True)
    expected_bootstrap = common.paired_class_bootstrap_many(
        evaluator.bootstrap_specs(rows),
        common.BOOTSTRAP_RESAMPLES,
        common.BOOTSTRAP_SEED,
    )
    saved_bootstrap = common.read_csv(output_dir / "paired_class_bootstrap.csv")
    compare_table(saved_bootstrap, expected_bootstrap, ("comparison",), tolerance=1e-9)

    expected_classwise = evaluator.classwise_summary(rows)
    saved_classwise = common.read_csv(output_dir / "classwise_summary.csv")
    compare_table(
        saved_classwise,
        expected_classwise,
        ("method", "target"),
        tolerance=1e-9,
    )

    expected_decomposition = evaluator.reference_quality_decomposition(rows)
    saved_decomposition = common.read_csv(
        output_dir / "reference_quality_decomposition.csv"
    )
    compare_table(
        saved_decomposition,
        expected_decomposition,
        ("group", "method"),
        tolerance=1e-9,
    )

    expected_gate = common.evaluate_clean_gate(expected_bootstrap)
    saved_gate = common.read_json(output_dir / "official_clean_gate.json")
    if saved_gate != expected_gate:
        raise AssertionError("Official clean gate does not reproduce")
    saved_gate_table = common.read_csv(output_dir / "preregistered_clean_gates.csv")
    compare_table(
        saved_gate_table,
        expected_gate["rules"],
        ("comparison",),
        tolerance=1e-9,
    )

    final = common.read_json(output_dir / "final_summary.json")
    if final.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Final protocol mismatch")
    if final.get("config_hash") != config_hash:
        raise AssertionError("Final config mismatch")
    if final.get("protocol_hash") != protocol_hash:
        raise AssertionError("Final preregistration mismatch")
    if final.get("selection_hash") != common.EXPECTED_SELECTION_HASH:
        raise AssertionError("Final selection mismatch")
    if final.get("status") != "PASS":
        raise AssertionError("Final status is not PASS")
    if final.get("official_clean_confirmation_completed") is not True:
        raise AssertionError("Final summary does not mark clean confirmation complete")
    if final.get("official_clean_summaries") != expected_summaries:
        raise AssertionError("Final summaries do not reproduce")
    if final.get("official_clean_paired_bootstrap") != expected_bootstrap:
        raise AssertionError("Final bootstrap does not reproduce")
    if final.get("official_clean_gate") != expected_gate:
        raise AssertionError("Final clean gate does not reproduce")
    if final.get("frozen_operating_point") != {
        "strength": {
            "candidate_id": "global_a1_s4_w4",
            "gate_mode": "global",
            "alpha_px": 1.0,
            "epsilon_px": 4.0,
            "steps": 4,
            "top1_weight": 1.0,
            "set_weight": 4.0,
        },
        "sigma": 0.025,
        "mc_draws": common.MC_DRAWS,
    }:
        raise AssertionError("Final frozen operating point differs")
    passed = bool(expected_gate["passed"])
    if bool(final.get("official_clean_confirmation_passed")) != passed:
        raise AssertionError("Final clean pass flag differs")
    if bool(final.get("attack_stage_ready")) != passed:
        raise AssertionError("Attack-stage gate differs")
    if bool(final.get("clean_paper_claim_authorized")) != passed:
        raise AssertionError("Clean-claim authorization differs")
    if final.get("adversarial_paper_claim_authorized") is not False:
        raise AssertionError("Adversarial claim must remain unauthorized before attack")
    if final.get("paper_claim_authorized") is not False:
        raise AssertionError("Paper claim must remain unauthorized before attack")
    expected_failed = [
        str(rule["comparison"])
        for rule in expected_gate["rules"]
        if not bool(rule["passed"])
    ]
    if final.get("failed_gates") != expected_failed:
        raise AssertionError("Final failed-gate list does not reproduce")
    expected_next = (
        "vit_reference_crossarch_adaptive_attack_pilot_on_attack_pilot_partition"
        if passed
        else "stop_and_report_failed_official_clean_confirmation"
    )
    if final.get("next_stage") != expected_next:
        raise AssertionError("Final next-stage decision differs")
    governance = final.get("data_governance", {})
    if governance.get("image_partitions_read") != [common.CLEAN_PARTITION]:
        raise AssertionError("Final data provenance differs")
    if governance.get("attack_outcomes_read") is not False:
        raise AssertionError("Final summary indicates attack-outcome access")
    if governance.get("calibration_images_read") is not False:
        raise AssertionError("Final summary indicates calibration-image reuse")
    if governance.get("no_selection") is not True:
        raise AssertionError("Final summary does not enforce no-selection")
    artifacts = final.get("artifacts", {})
    if artifacts.get("per_example_file") != per_example_path.name:
        raise AssertionError("Final per-example filename differs")
    if artifacts.get("per_example_file_sha256") != common.file_sha256(
        per_example_path
    ):
        raise AssertionError("Final per-example checksum differs")
    if int(artifacts.get("per_example_rows", -1)) != per_example_count:
        raise AssertionError("Final per-example row count differs")
    if int(artifacts.get("partial_files", -1)) != common.CLEAN_SHARDS:
        raise AssertionError("Final partial-file count differs")
    expected_wall_seconds = sum(float(row["wall_seconds"]) for row in runtime_rows)
    assert_close(
        float(artifacts.get("partial_wall_seconds_sum", -1.0)),
        expected_wall_seconds,
        1e-9,
        "Final partial runtime sum",
    )
    if int(artifacts.get("classwise_rows", -1)) != len(expected_classwise):
        raise AssertionError("Final classwise row count differs")

    table = {str(row["method"]): row for row in expected_summaries}
    print(
        f"PASS {output_dir}: {common.CLEAN_EXAMPLES:,} clean_confirmation images, "
        f"{common.CLEAN_SHARDS} resumable shards, MC={common.MC_DRAWS}, paired CIs reproduce"
    )
    for method in common.METHODS:
        row = table[method]
        print(
            f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Top5={float(row['label_coverage_k5_probability_percent']):.3f}% "
            f"Exact5={float(row['exact_set_k5_probability_percent']):.3f}% "
            f"Exact2={float(row['exact_set_k2_probability_percent']):.3f}%"
        )
    print("OFFICIAL_CLEAN_CONFIRMATION=" + ("PASS" if passed else "FAIL"))
    print("ATTACK_STAGE_READY=" + ("YES" if passed else "NO"))
    print("PAPER_CLAIM_AUTHORIZED=NO")
    print(
        "ImageNet ViT-reference VR2 audit complete; no training, attack, download, "
        "calibration image, or attack-partition image was read."
    )


if __name__ == "__main__":
    main()
