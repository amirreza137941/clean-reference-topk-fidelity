#!/usr/bin/env python3
"""Independent structural/statistical auditor for ImageNet R3 clean results."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import evaluate_topk_imagenet_clean_confirmation_r3 as evaluator
import topk_imagenet_clean_confirmation_r3_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ImageNet clean confirmation R3")
    parser.add_argument("--output-dir", type=Path, default=evaluator.DEFAULT_OUTPUT_DIR)
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
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"Not a Boolean: {value!r}")


def equivalent(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    if isinstance(right, bool):
        try:
            return as_bool(left) == right
        except ValueError:
            return False
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    return math.isclose(left_float, right_float, rel_tol=0.0, abs_tol=tolerance)


def compare_row_sets(
    saved: Sequence[Mapping[str, Any]],
    recomputed: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
    label: str,
    tolerance: float = 1e-10,
) -> None:
    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in key_fields)

    saved_map = {key(row): row for row in saved}
    recomputed_map = {key(row): row for row in recomputed}
    if len(saved_map) != len(saved) or len(recomputed_map) != len(recomputed):
        raise AssertionError(f"Duplicate keys in {label}")
    if set(saved_map) != set(recomputed_map):
        raise AssertionError(f"Key mismatch in {label}")
    for row_key in sorted(saved_map):
        left = saved_map[row_key]
        right = recomputed_map[row_key]
        if set(left) != set(right):
            raise AssertionError(f"Column mismatch in {label}/{row_key}")
        for field in right:
            if not equivalent(left[field], right[field], tolerance):
                raise AssertionError(
                    f"Value mismatch in {label}/{row_key}/{field}: "
                    f"saved={left[field]!r}, recomputed={right[field]!r}"
                )


def verify_protocol_and_config(output_dir: Path) -> tuple[dict[str, Any], str]:
    protocol = common.read_json(output_dir / "frozen_protocol_r3.json")
    expected_protocol = common.frozen_protocol_payload()
    if protocol != expected_protocol:
        raise AssertionError("Frozen R3 preregistration changed")
    protocol_hash = str(protocol["protocol_hash"])

    config = common.read_json(output_dir / "run_config.json")
    if config.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("R3 config protocol mismatch")
    if config.get("protocol_hash") != protocol_hash:
        raise AssertionError("R3 config preregistration mismatch")
    working = dict(config)
    config_hash = str(working.pop("config_hash"))
    if common.object_sha256(working) != config_hash:
        raise AssertionError("R3 config hash mismatch")
    if config.get("partition") != common.PARTITION:
        raise AssertionError("R3 config partition mismatch")
    if int(config.get("examples")) != common.EXAMPLES:
        raise AssertionError("R3 config example count changed")
    if int(config.get("mc_draws")) != common.MC_DRAWS:
        raise AssertionError("R3 MC count changed")
    if config.get("frozen_strength") != common.FROZEN_STRENGTH:
        raise AssertionError("R3 strength changed")
    if abs(float(config.get("sigma")) - common.SIGMA) > 1e-12:
        raise AssertionError("R3 sigma changed")
    if config.get("selection_or_retuning_allowed") is not False:
        raise AssertionError("R3 improperly permits selection")
    if config.get("attack_allowed") is not False:
        raise AssertionError("R3 improperly permits attack")
    if config.get("partitions_read") != [common.PARTITION]:
        raise AssertionError("R3 read an unexpected partition")
    for name, claimed in config.get("script_sha256", {}).items():
        path = evaluator.PROJECT_DIR / name
        if not path.is_file() or common.file_sha256(path) != claimed:
            raise AssertionError(f"Source changed since R3 evaluation: {name}")
    expected_sources = set(common.NEW_SCRIPT_NAMES) | set(common.EXPECTED_R2_SOURCE_HASHES)
    if set(config.get("script_sha256", {})) != expected_sources:
        raise AssertionError("R3 source manifest is incomplete")
    return config, config_hash


def verify_manifest(
    output_dir: Path, holdout_rows: Sequence[Mapping[str, object]], protocol_hash: str
) -> tuple[list[list[dict[str, object]]], dict[str, dict[str, Any]]]:
    manifest = common.read_json(output_dir / "clean_confirmation_manifest.json")
    if manifest.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("R3 manifest protocol mismatch")
    if manifest.get("protocol_hash") != protocol_hash:
        raise AssertionError("R3 manifest preregistration mismatch")
    if manifest.get("partition") != common.PARTITION:
        raise AssertionError("R3 manifest partition mismatch")
    if int(manifest.get("examples")) != common.EXAMPLES:
        raise AssertionError("R3 manifest example count mismatch")
    if manifest.get("partitions_read") != [common.PARTITION]:
        raise AssertionError("R3 manifest read an unexpected partition")
    if manifest.get("attack_outcomes_read") is not False:
        raise AssertionError("R3 manifest indicates attack-outcome access")

    shards = evaluator.outcome_blind_shards(holdout_rows)
    expected_rows: list[dict[str, Any]] = []
    for index, shard_rows in enumerate(shards):
        for row in shard_rows:
            expected_rows.append({**dict(row), "shard_id": common.shard_id(index)})
    if common.object_sha256(expected_rows) != manifest.get("rows_sha256"):
        raise AssertionError("R3 manifest row hash does not reproduce")
    saved_rows = common.read_csv(output_dir / "clean_confirmation_manifest.csv")
    if len(saved_rows) != common.EXAMPLES:
        raise AssertionError("R3 manifest CSV row count mismatch")
    expected_by_path = {str(row["relative_path"]): row for row in expected_rows}
    saved_by_path = {str(row["relative_path"]): row for row in saved_rows}
    if len(saved_by_path) != common.EXAMPLES or set(saved_by_path) != set(expected_by_path):
        raise AssertionError("R3 manifest membership mismatch")
    for path, expected in expected_by_path.items():
        saved = saved_by_path[path]
        for field in ("partition", "target", "class_name", "within_class_slot", "shard_id"):
            if not equivalent(saved[field], expected[field]):
                raise AssertionError(f"R3 manifest mismatch for {path}/{field}")
    return shards, expected_by_path


def verify_method_row(row: Mapping[str, Any], method: str) -> None:
    specs = evaluator.defense_specs()
    expected = specs[method]
    exact_fields = {
        "gate_mode": expected.gate_mode,
        "gate_active": int(expected.gate_active),
        "draws": expected.draws,
        "sigma": expected.sigma,
        "strength_alpha_px": expected.strength_alpha_px,
        "strength_epsilon_px": expected.strength_epsilon_px,
        "strength_steps": expected.strength_steps,
        "top1_weight": expected.top1_weight,
        "set_weight": expected.set_weight,
    }
    for field, value in exact_fields.items():
        if not equivalent(row[field], value):
            raise AssertionError(f"Frozen method specification mismatch: {method}/{field}")
    draws = int(row["draws"])
    for field in common.ALL_PROBABILITY_KEYS:
        value = float(row[field])
        if not -1e-12 <= value <= 1.0 + 1e-12:
            raise AssertionError(f"Probability outside [0,1]: {method}/{field}")
        if abs(value * draws - round(value * draws)) > 1e-8:
            raise AssertionError(f"Probability is not a {draws}-draw empirical fraction")
    for field in (
        "mean_logits_top1_accuracy",
        "mean_logits_label_coverage_k5",
        "mean_logits_exact_set_k5",
        "mean_logits_exact_set_k2",
    ):
        if float(row[field]) not in (0.0, 1.0):
            raise AssertionError(f"Mean-logit indicator is non-binary: {method}/{field}")
    if method == "standard":
        for field in (
            "reference_top1_match_probability",
            "exact_set_k5_probability",
            "exact_set_k2_probability",
        ):
            if float(row[field]) != 1.0:
                raise AssertionError(f"Standard-reference identity failed for {field}")


def verify_partials(
    output_dir: Path,
    shards: Sequence[Sequence[Mapping[str, object]]],
    config_hash: str,
    protocol_hash: str,
    manifest_by_path: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, shard_rows in enumerate(shards):
        shard = common.shard_id(index)
        path = evaluator.partial_path(output_dir, shard)
        if not path.is_file():
            raise FileNotFoundError(f"Missing R3 partial: {path}")
        payload = common.read_json(path)
        payloads.append(payload)
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("protocol_hash") != protocol_hash:
            raise AssertionError(f"Partial protocol hash mismatch: {path}")
        if payload.get("config_hash") != config_hash:
            raise AssertionError(f"Partial config hash mismatch: {path}")
        if payload.get("r2_selection_hash") != common.R2_SELECTION_HASH:
            raise AssertionError(f"Partial selection hash mismatch: {path}")
        if payload.get("partition") != common.PARTITION or payload.get("shard_id") != shard:
            raise AssertionError(f"Partial shard/partition mismatch: {path}")
        if int(payload.get("input_examples")) != common.EXAMPLES_PER_SHARD:
            raise AssertionError(f"Partial input count mismatch: {path}")
        rows = [dict(row) for row in payload.get("examples", [])]
        if len(rows) != common.EXAMPLES_PER_SHARD * len(common.METHODS):
            raise AssertionError(f"Partial method-row count mismatch: {path}")
        expected_paths = sorted(str(row["relative_path"]) for row in shard_rows)
        counts = Counter(str(row["method"]) for row in rows)
        if counts != Counter({method: common.EXAMPLES_PER_SHARD for method in common.METHODS}):
            raise AssertionError(f"Partial method coverage mismatch: {path}")
        method_paths = {
            method: sorted(str(row["relative_path"]) for row in rows if row["method"] == method)
            for method in common.METHODS
        }
        if any(paths != expected_paths for paths in method_paths.values()):
            raise AssertionError(f"Partial sample alignment mismatch: {path}")
        anchors: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            method = str(row["method"])
            relative_path = str(row["relative_path"])
            pair = (method, relative_path)
            if pair in seen:
                raise AssertionError(f"Duplicate method/sample row: {pair}")
            seen.add(pair)
            if row.get("protocol_id") != common.PROTOCOL_ID:
                raise AssertionError(f"Example protocol mismatch: {pair}")
            if row.get("partition") != common.PARTITION or row.get("shard_id") != shard:
                raise AssertionError(f"Example provenance mismatch: {pair}")
            if relative_path not in manifest_by_path:
                raise AssertionError(f"Example outside frozen manifest: {relative_path}")
            if int(row["label"]) != int(manifest_by_path[relative_path]["target"]):
                raise AssertionError(f"Example label mismatch: {relative_path}")
            anchor = (
                int(row["label"]),
                int(row["reference_top1"]),
                str(row["reference_set_k5"]),
                str(row["reference_set_k2"]),
            )
            if relative_path in anchors and anchors[relative_path] != anchor:
                raise AssertionError(f"Cross-method reference mismatch: {relative_path}")
            anchors[relative_path] = anchor
            verify_method_row(row, method)
        all_rows.extend(rows)
    if len(seen) != common.EXAMPLES * len(common.METHODS):
        raise AssertionError("R3 combined method/sample coverage is incomplete")
    return all_rows, payloads


def verify_saved_analysis(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    payloads: Sequence[Mapping[str, Any]],
    config_hash: str,
    protocol_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    summaries = evaluator.summarize_rows(rows)
    saved_summaries = common.read_csv(output_dir / "clean_confirmation_summary.csv")
    compare_row_sets(saved_summaries, summaries, ("method",), "summary")

    bootstrap = common.paired_class_bootstrap_many(
        evaluator.build_bootstrap_specs(rows),
        common.BOOTSTRAP_RESAMPLES,
        common.BOOTSTRAP_SEED,
    )
    saved_bootstrap = common.read_csv(output_dir / "paired_class_bootstrap.csv")
    compare_row_sets(
        saved_bootstrap, bootstrap, ("comparison_id",), "paired class bootstrap", 1e-9
    )

    gates = common.apply_preregistered_gates(bootstrap)
    saved_gates = common.read_csv(output_dir / "preregistered_clean_gates.csv")
    compare_row_sets(saved_gates, gates, ("gate_id",), "preregistered gates", 1e-9)

    decomposition = evaluator.reference_quality_decomposition(rows)
    saved_decomposition = common.read_csv(output_dir / "reference_quality_decomposition.csv")
    compare_row_sets(
        saved_decomposition,
        decomposition,
        ("subset", "method"),
        "reference-quality decomposition",
    )
    classwise = evaluator.classwise_summary(rows)
    saved_classwise = common.read_csv(output_dir / "classwise_summary.csv")
    compare_row_sets(
        saved_classwise, classwise, ("method", "target"), "classwise summary"
    )

    final = common.read_json(output_dir / "final_summary.json")
    if final.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError("Final-summary protocol mismatch")
    if final.get("protocol_hash") != protocol_hash or final.get("config_hash") != config_hash:
        raise AssertionError("Final-summary invocation mismatch")
    if final.get("r2_selection_hash") != common.R2_SELECTION_HASH:
        raise AssertionError("Final-summary selection mismatch")
    if final.get("status") != "PASS":
        raise AssertionError("R3 did not complete with structural PASS")
    clean_pass = all(bool(row["passed"]) for row in gates)
    failed = [str(row["gate_id"]) for row in gates if not bool(row["passed"])]
    if final.get("official_clean_confirmation") != ("PASS" if clean_pass else "FAIL"):
        raise AssertionError("Final clean-confirmation decision mismatch")
    if bool(final.get("official_clean_stage_passed")) != clean_pass:
        raise AssertionError("Final clean-stage Boolean mismatch")
    if bool(final.get("attack_stage_ready")) != clean_pass:
        raise AssertionError("Final attack-stage decision mismatch")
    if bool(final.get("clean_paper_claim_authorized")) != clean_pass:
        raise AssertionError("Final clean-claim decision mismatch")
    if final.get("adversarial_paper_claim_authorized") is not False:
        raise AssertionError("R3 cannot authorize an adversarial ImageNet claim")
    if final.get("failed_gates") != failed:
        raise AssertionError("Final failed-gate list mismatch")

    per_example = output_dir / "clean_confirmation_per_example.csv.gz"
    if common.file_sha256(per_example) != final["artifacts"]["per_example_file_sha256"]:
        raise AssertionError("Compressed per-example artifact hash mismatch")
    count = 0
    for row in common.iter_csv(per_example):
        count += 1
        if row.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError("Per-example CSV contains another protocol")
    if count != common.EXAMPLES * len(common.METHODS):
        raise AssertionError("Compressed per-example row count mismatch")
    if int(final["artifacts"]["per_example_rows"]) != count:
        raise AssertionError("Final per-example count mismatch")
    if int(final["artifacts"]["partial_files"]) != len(payloads):
        raise AssertionError("Final partial count mismatch")
    return summaries, gates, clean_pass


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": resolved(args.data_root),
        "r0_dir": resolved(args.r0_dir),
        "r1_dir": resolved(args.r1_dir),
        "r2_dir": resolved(args.r2_dir),
        "standard_checkpoint": resolved(args.standard_checkpoint),
        "robust_checkpoint": resolved(args.robust_checkpoint),
        "output_dir": resolved(args.output_dir),
    }
    output_dir = paths["output_dir"]
    config, config_hash = verify_protocol_and_config(output_dir)
    del config
    holdout_rows, _ = evaluator.validate_parents(paths)
    protocol_hash = str(common.frozen_protocol_payload()["protocol_hash"])
    shards, manifest_by_path = verify_manifest(output_dir, holdout_rows, protocol_hash)
    rows, payloads = verify_partials(
        output_dir, shards, config_hash, protocol_hash, manifest_by_path
    )
    summaries, gates, clean_pass = verify_saved_analysis(
        output_dir, rows, payloads, config_hash, protocol_hash
    )

    table = {str(row["method"]): row for row in summaries}
    print(
        f"PASS {output_dir}: {common.SHARDS} shards, "
        f"{len(rows):,} method rows, 1,000 class clusters",
        flush=True,
    )
    for method in common.METHODS:
        row = table[method]
        print(
            f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Top5={float(row['label_coverage_k5_probability_percent']):.3f}% "
            f"Exact5={float(row['exact_set_k5_probability_percent']):.3f}% "
            f"Exact2={float(row['exact_set_k2_probability_percent']):.3f}%",
            flush=True,
        )
    print("Preregistered clean gates", flush=True)
    for row in gates:
        print(
            f"  {row['gate_id']}: {float(row['difference_pp']):+.3f} pp "
            f"[{float(row['ci95_low_pp']):+.3f}, {float(row['ci95_high_pp']):+.3f}] "
            f"-> {row['status']}",
            flush=True,
        )
    print("OFFICIAL_CLEAN_CONFIRMATION=" + ("PASS" if clean_pass else "FAIL"))
    print("ATTACK_STAGE_READY=" + ("YES" if clean_pass else "NO"))
    print("ImageNet R3 audit complete; no training, retuning, attack, merge, or deletion was performed.")


if __name__ == "__main__":
    main()
