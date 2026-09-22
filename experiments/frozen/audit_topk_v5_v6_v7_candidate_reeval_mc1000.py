#!/usr/bin/env python3
"""Fail-fast audit for the combined V5/V6/V7 common-MC=1000 results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from reevaluate_topk_v7_candidates_mc1000 import (
    COMBINED_SOURCE_IDS,
    EXPECTED_BOOTSTRAP_RESAMPLES,
    EXPECTED_BOOTSTRAP_SEED,
    EXPECTED_BUDGET,
    EXPECTED_COMMON_DRAWS,
    EXPECTED_COMMON_SEED,
    EXPECTED_COMPARISON_COUNT,
    EXPECTED_DATA_SEED,
    EXPECTED_ENGINE_PROTOCOL,
    EXPECTED_ENGINE_SHA256,
    EXPECTED_EPSILON_PX,
    EXPECTED_EXAMPLES,
    EXPECTED_K,
    EXPECTED_PRIOR_PROTOCOL,
    EXPECTED_PRIOR_SCRIPT_SHA256,
    EXPECTED_V7_PROTOCOL,
    FLOAT32_AGGREGATE_TOLERANCE_PP,
    PRIOR_SOURCE_IDS,
    REEVAL_PROTOCOL,
    V7_SEEDS,
    V7_SOURCE_IDS,
    attack_envelope,
    build_comparisons,
    read_csv,
    read_json,
    sha256_file,
)
from topk_v7_candidate_reeval_launcher_common import OUTPUT_FULL, OUTPUT_NOISE


ROBUST_MARGIN_NUMERICAL_TOLERANCE = 5e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, default=OUTPUT_FULL)
    parser.add_argument("--noise-dir", type=Path, default=OUTPUT_NOISE)
    return parser.parse_args()


def require_files(directory: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise AssertionError(f"{directory} is missing: {', '.join(missing)}")


def close(a: float, b: float, tolerance: float = 2e-6) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def assert_original_fields(
    original: Mapping[str, str],
    combined: Mapping[str, str],
    *,
    context: str,
) -> None:
    mismatches = {
        key: (combined.get(key), value)
        for key, value in original.items()
        if combined.get(key) != value
    }
    if mismatches:
        preview = dict(list(mismatches.items())[:5])
        raise AssertionError(f"Reused prior row changed in {context}: {preview}")


def compare_csv_rows_exact(
    left: Sequence[Mapping[str, str]],
    right: Sequence[Mapping[str, str]],
    *,
    context: str,
) -> None:
    if len(left) != len(right):
        raise AssertionError(f"Row count differs in {context}")
    for position, (a, b) in enumerate(zip(left, right)):
        if a != b:
            keys = sorted(set(a) | set(b))
            mismatches = {
                key: (a.get(key), b.get(key))
                for key in keys
                if a.get(key) != b.get(key)
            }
            raise AssertionError(
                f"CSV row differs in {context}, row={position}: "
                f"{dict(list(mismatches.items())[:5])}"
            )


def audit_mode(directory: Path, mode: str):
    directory = directory.expanduser().resolve()
    required = (
        "run_config_candidate_reeval_v7.json",
        "candidate_reeval_v5_v6_v7_summary.csv",
        "candidate_reeval_v5_v6_v7_per_example.csv",
        "candidate_reeval_v5_v6_v7_comparisons.csv",
        "candidate_reeval_v5_v6_v7_summary.json",
    )
    require_files(directory, required)
    config = read_json(directory / "run_config_candidate_reeval_v7.json")
    expected_batch = 5 if mode == "full" else 25
    expected_config = {
        "protocol_version": REEVAL_PROTOCOL,
        "mode": mode,
        "batch_size": expected_batch,
        "eval_draws": EXPECTED_COMMON_DRAWS,
        "reeval_seed": EXPECTED_COMMON_SEED,
        "bootstrap_resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
        "data_seed": EXPECTED_DATA_SEED,
        "k": EXPECTED_K,
        "epsilon_px": EXPECTED_EPSILON_PX,
        "source_attack_budget": EXPECTED_BUDGET,
        "candidate_source_ids": list(COMBINED_SOURCE_IDS),
        "reused_source_ids": list(PRIOR_SOURCE_IDS),
        "newly_evaluated_source_ids": list(V7_SOURCE_IDS),
        "common_random_numbers": True,
        "reeval_draws_counted_in_attack_budget": False,
        "prior_protocol_version": EXPECTED_PRIOR_PROTOCOL,
        "prior_helper_sha256": EXPECTED_PRIOR_SCRIPT_SHA256,
        "evaluator_sha256": EXPECTED_ENGINE_SHA256,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"{directory} config mismatch: {mismatches}")
    anchor = config.get("robust_anchor_validation", {})
    if anchor.get("discrete_outputs_exactly_reproduced") is not True:
        raise AssertionError("Deterministic robust preflight was not recorded as exact")
    if int(anchor.get("actual_draws", -1)) != 1:
        raise AssertionError("Robust preflight must use one deterministic draw")

    project_dir = Path(__file__).resolve().parent
    local_hashes = {
        "script_sha256": sha256_file(
            project_dir / "reevaluate_topk_v7_candidates_mc1000.py"
        ),
        "prior_helper_sha256": sha256_file(
            project_dir / "reevaluate_topk_v5_v6_candidates_mc1000.py"
        ),
        "evaluator_sha256": sha256_file(
            project_dir / "evaluate_general_topk_defense_v6_cifar10.py"
        ),
        "standard_weights_sha256": sha256_file(
            Path(config["standard_weights"]).expanduser().resolve()
        ),
        "dual_weights_sha256": sha256_file(
            Path(config["dual_weights"]).expanduser().resolve()
        ),
    }
    for key, observed in local_hashes.items():
        if config.get(key) != observed:
            raise AssertionError(f"Current artifact hash differs for {key}")

    prior_dir = Path(config["prior_r2_dir"]).expanduser().resolve()
    prior_files = {
        "prior_run_config_sha256": prior_dir / "run_config_candidate_reeval.json",
        "prior_summary_csv_sha256": prior_dir / "candidate_reeval_summary.csv",
        "prior_per_example_csv_sha256": prior_dir / "candidate_reeval_per_example.csv",
    }
    for key, path in prior_files.items():
        if not path.is_file() or sha256_file(path) != config.get(key):
            raise AssertionError(f"Prior R2 provenance changed for {key}")

    summaries = read_csv(directory / "candidate_reeval_v5_v6_v7_summary.csv")
    examples = read_csv(directory / "candidate_reeval_v5_v6_v7_per_example.csv")
    comparisons = read_csv(directory / "candidate_reeval_v5_v6_v7_comparisons.csv")
    final = read_json(directory / "candidate_reeval_v5_v6_v7_summary.json")
    if len(summaries) != len(COMBINED_SOURCE_IDS):
        raise AssertionError(f"Expected {len(COMBINED_SOURCE_IDS)} summary rows")
    if len(examples) != len(COMBINED_SOURCE_IDS) * EXPECTED_EXAMPLES:
        raise AssertionError("Combined per-example grid is incomplete")
    if len(comparisons) != EXPECTED_COMPARISON_COUNT:
        raise AssertionError("Comparison grid is incomplete")
    if tuple(row["source_id"] for row in summaries) != COMBINED_SOURCE_IDS:
        raise AssertionError("Combined source ordering or IDs differ")

    summary_by_id = {row["source_id"]: row for row in summaries}
    examples_by_id: dict[str, list[dict[str, str]]] = {}
    for row in examples:
        examples_by_id.setdefault(row["source_id"], []).append(row)
    if set(examples_by_id) != set(COMBINED_SOURCE_IDS):
        raise AssertionError("Combined per-example source IDs are incomplete")

    canonical_key: list[tuple[str, str, str, str]] | None = None
    for source_id in COMBINED_SOURCE_IDS:
        summary = summary_by_id[source_id]
        rows = sorted(examples_by_id[source_id], key=lambda row: int(row["position"]))
        if len(rows) != EXPECTED_EXAMPLES:
            raise AssertionError(f"Wrong row count for {source_id}")
        if [int(row["position"]) for row in rows] != list(range(EXPECTED_EXAMPLES)):
            raise AssertionError(f"Invalid positions for {source_id}")
        if len({row["sample_id"] for row in rows}) != EXPECTED_EXAMPLES:
            raise AssertionError(f"Duplicate samples for {source_id}")
        if int(summary["reeval_seed"]) != EXPECTED_COMMON_SEED:
            raise AssertionError(f"Wrong common seed for {source_id}")
        if int(summary["batch_size"]) != expected_batch:
            raise AssertionError(f"Wrong batch size for {source_id}")
        if summary["common_random_numbers"] != "True":
            raise AssertionError(f"Missing common-RNG flag for {source_id}")
        if summary["reeval_draws_counted_in_attack_budget"] != "False":
            raise AssertionError("Measurement draws were charged to attack budget")
        if re.fullmatch(r"[0-9a-f]{64}", summary["source_candidate_sha256"]) is None:
            raise AssertionError(f"Invalid candidate hash for {source_id}")
        defense = summary["source_defense"]
        expected_draws = EXPECTED_COMMON_DRAWS if defense == "proposed" else 1
        if int(summary["reeval_draws_actual"]) != expected_draws:
            raise AssertionError(f"Wrong draw count for {source_id}")
        if any(int(row["eval_draws"]) != expected_draws for row in rows):
            raise AssertionError(f"Per-example draw count mismatch for {source_id}")
        if any(int(row["reeval_seed"]) != EXPECTED_COMMON_SEED for row in rows):
            raise AssertionError(f"Per-example seed mismatch for {source_id}")
        if any(
            row["source_candidate_sha256"] != summary["source_candidate_sha256"]
            for row in rows
        ):
            raise AssertionError(f"Candidate hash mismatch in rows for {source_id}")
        key = [
            (
                row["sample_id"],
                row["sample_index"],
                row["true_label"],
                row["reference_set"],
            )
            for row in rows
        ]
        if canonical_key is None:
            canonical_key = key
        elif key != canonical_key:
            raise AssertionError(f"Sample/reference mismatch for {source_id}")
        exact = np.array([float(row["exact_set_probability"]) for row in rows])
        aggregate = 100.0 * float(exact.mean())
        if not close(
            aggregate,
            float(summary["exact_set"]),
            tolerance=FLOAT32_AGGREGATE_TOLERANCE_PP,
        ):
            raise AssertionError(
                f"Exact-set aggregate does not match rows for {source_id}: "
                f"{summary['exact_set']} vs {aggregate}"
            )
        if defense == "proposed":
            grid_error = np.max(
                np.abs(exact * EXPECTED_COMMON_DRAWS - np.round(exact * EXPECTED_COMMON_DRAWS))
            )
            if grid_error > 2e-4:
                raise AssertionError(f"{source_id} is not on the MC=1000 grid")
        if source_id in V7_SOURCE_IDS:
            if summary["source_version"] != "V7":
                raise AssertionError(f"Wrong version for {source_id}")
            if summary["source_protocol"] != EXPECTED_V7_PROTOCOL:
                raise AssertionError(f"Wrong V7 experiment protocol for {source_id}")
            if summary["source_engine_protocol"] != EXPECTED_ENGINE_PROTOCOL:
                raise AssertionError(f"Wrong V7 engine protocol for {source_id}")
            if summary["source_paper_attack"] != "SquareHybrid":
                raise AssertionError(f"Wrong V7 paper label for {source_id}")
            if summary["clean_tensor_source"] != "stored_in_candidate":
                raise AssertionError(f"Wrong clean provenance for {source_id}")
            if summary["measurement_origin"] != "new_v7_common_mc1000":
                raise AssertionError(f"Wrong measurement origin for {source_id}")
        elif summary["measurement_origin"] != "reused_prior_r2":
            raise AssertionError(f"Wrong prior provenance for {source_id}")

    # Every reused field must be byte-for-byte identical after CSV parsing.
    prior_summaries = read_csv(prior_dir / "candidate_reeval_summary.csv")
    prior_examples = read_csv(prior_dir / "candidate_reeval_per_example.csv")
    for row in prior_summaries:
        assert_original_fields(
            row,
            summary_by_id[row["source_id"]],
            context=f"summary/{row['source_id']}",
        )
    prior_by_id: dict[str, list[dict[str, str]]] = {}
    for row in prior_examples:
        prior_by_id.setdefault(row["source_id"], []).append(row)
    for source_id in PRIOR_SOURCE_IDS:
        original = sorted(prior_by_id[source_id], key=lambda row: int(row["position"]))
        combined = sorted(
            examples_by_id[source_id], key=lambda row: int(row["position"])
        )
        for position, (left, right) in enumerate(zip(original, combined)):
            assert_original_fields(
                left,
                right,
                context=f"per-example/{source_id}/{position}",
            )

    # V7 combined rows must exactly match their resume-safe partials.
    for source_id in V7_SOURCE_IDS:
        partial_summary = read_json(directory / f"partials/{source_id}.summary.json")
        partial_examples = read_csv(directory / f"partials/{source_id}.per_example.csv")
        combined_summary = {
            key: value for key, value in summary_by_id[source_id].items() if value != ""
        }
        for key, value in partial_summary.items():
            observed = combined_summary.get(key)
            if isinstance(value, bool):
                expected = str(value)
            else:
                expected = str(value)
            if observed != expected:
                raise AssertionError(
                    f"V7 partial summary mismatch for {source_id}, field={key}: "
                    f"{observed!r} != {expected!r}"
                )
        compare_csv_rows_exact(
            partial_examples,
            sorted(examples_by_id[source_id], key=lambda row: int(row["position"])),
            context=f"V7 partial/{source_id}",
        )

    recalculated = build_comparisons(
        summaries,
        examples,
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    if [row["comparison"] for row in comparisons] != [
        row["comparison"] for row in recalculated
    ]:
        raise AssertionError("Comparison ordering or names differ")
    numeric_fields = (
        "baseline_exact_set_pct",
        "candidate_exact_set_pct",
        "candidate_minus_baseline_pp",
        "ci95_low_pp",
        "ci95_high_pp",
    )
    exact_fields = (
        "comparison_family",
        "baseline_source_id",
        "candidate_source_id",
        "attack_seed",
        "positive_delta_means",
        "verdict",
        "verdict_label",
        "inference_scope",
        "baseline_seed_count",
        "candidate_seed_count",
        "bootstrap_resamples",
        "bootstrap_seed",
    )
    for stored, observed in zip(comparisons, recalculated):
        for key in numeric_fields:
            if not close(float(stored[key]), float(observed[key])):
                raise AssertionError(
                    f"Comparison numeric mismatch: {stored['comparison']} {key}"
                )
        for key in exact_fields:
            if str(stored[key]) != str(observed[key]):
                raise AssertionError(
                    f"Comparison provenance mismatch: {stored['comparison']} {key}"
                )

    expected_envelope = attack_envelope(summaries)
    if final.get("attack_envelope") != expected_envelope:
        raise AssertionError("Final attack envelope does not match the summary rows")
    expected_final = {
        "protocol_version": REEVAL_PROTOCOL,
        "mode": mode,
        "candidate_count": len(COMBINED_SOURCE_IDS),
        "new_candidate_count": len(V7_SOURCE_IDS),
        "per_example_rows": len(COMBINED_SOURCE_IDS) * EXPECTED_EXAMPLES,
        "comparison_count": EXPECTED_COMPARISON_COUNT,
        "common_reeval_seed": EXPECTED_COMMON_SEED,
        "requested_mc_draws": EXPECTED_COMMON_DRAWS,
    }
    for key, value in expected_final.items():
        if final.get(key) != value:
            raise AssertionError(f"Final JSON mismatch for {key}")
    expected_values = {
        source_id: float(summary_by_id[source_id]["exact_set"])
        for source_id in COMBINED_SOURCE_IDS
    }
    if final.get("source_exact_set_pct") != expected_values:
        raise AssertionError("Final JSON Exact-set mapping differs")
    expected_hashes = {
        source_id: summary_by_id[source_id]["source_candidate_sha256"]
        for source_id in COMBINED_SOURCE_IDS
    }
    if final.get("source_candidate_sha256") != expected_hashes:
        raise AssertionError("Final JSON candidate hashes differ")

    v5 = float(summary_by_id["v5_proposed_strongest"]["exact_set"])
    v6 = [
        float(summary_by_id[f"v6_proposed_seed_{seed}"]["exact_set"])
        for seed in V7_SEEDS
    ]
    v7 = [
        float(summary_by_id[f"v7_proposed_seed_{seed}"]["exact_set"])
        for seed in V7_SEEDS
    ]
    v7_v5 = next(
        row
        for row in comparisons
        if row["comparison"] == "v7_three_seed_mean_minus_v5_proposed_strongest"
    )
    v7_v6 = next(
        row
        for row in comparisons
        if row["comparison"] == "v7_three_seed_mean_minus_v6_three_seed_mean"
    )
    print(
        f"PASS {directory}: V5={v5:.4f}%, "
        f"V6={mean(v6):.4f}±{stdev(v6):.4f}%, "
        f"V7={mean(v7):.4f}±{stdev(v7):.4f}%",
        flush=True,
    )
    print(
        f"  V7mean−V5={float(v7_v5['candidate_minus_baseline_pp']):+.4f} pp "
        f"[{float(v7_v5['ci95_low_pp']):+.4f}, "
        f"{float(v7_v5['ci95_high_pp']):+.4f}]",
        flush=True,
    )
    print(
        f"  V7mean−V6mean={float(v7_v6['candidate_minus_baseline_pp']):+.4f} pp "
        f"[{float(v7_v6['ci95_low_pp']):+.4f}, "
        f"{float(v7_v6['ci95_high_pp']):+.4f}]",
        flush=True,
    )
    print(
        "  envelope="
        f"{expected_envelope['overall_strongest_observed_source_id']} "
        f"({expected_envelope['overall_attack_envelope_exact_set_pct']:.4f}%)",
        flush=True,
    )
    return summary_by_id, examples_by_id, comparisons, final


def compare_modes(full, noise) -> None:
    full_summaries, full_examples, _, _ = full
    noise_summaries, noise_examples, _, _ = noise
    reference_id = "v5_proposed_strongest"
    full_reference = sorted(
        full_examples[reference_id], key=lambda row: int(row["position"])
    )
    noise_reference = sorted(
        noise_examples[reference_id], key=lambda row: int(row["position"])
    )
    fields = ("position", "sample_id", "sample_index", "true_label", "reference_set")
    for left, right in zip(full_reference, noise_reference):
        if any(left[field] != right[field] for field in fields):
            raise AssertionError("Full and Noise-only use different fixed subsets")

    robust_ids = [source_id for source_id in PRIOR_SOURCE_IDS if "robust" in source_id]
    max_margin_drift = 0.0
    for source_id in robust_ids:
        if full_summaries[source_id]["exact_set"] != noise_summaries[source_id]["exact_set"]:
            raise AssertionError(f"Robust aggregate differs across modes for {source_id}")
        left_rows = sorted(
            full_examples[source_id], key=lambda row: int(row["position"])
        )
        right_rows = sorted(
            noise_examples[source_id], key=lambda row: int(row["position"])
        )
        for left, right in zip(left_rows, right_rows):
            for field in (
                "sample_id",
                "reference_set",
                "exact_set_probability",
                "top1_probability",
                "coverage_probability",
            ):
                if left[field] != right[field]:
                    raise AssertionError(
                        f"Robust discrete output differs across modes for {source_id}"
                    )
            max_margin_drift = max(
                max_margin_drift,
                abs(float(left["boundary_margin"]) - float(right["boundary_margin"])),
            )
    if max_margin_drift > ROBUST_MARGIN_NUMERICAL_TOLERANCE:
        raise AssertionError(f"Robust continuous margin drift too large: {max_margin_drift}")
    print(
        "PASS Full/Noise-only share samples, labels, and references; prior robust "
        f"anchors remain reproducible (max margin drift={max_margin_drift:.6g})."
    )


def main() -> None:
    args = parse_args()
    full = audit_mode(args.full_dir, "full")
    noise = audit_mode(args.noise_dir, "noise_only")
    compare_modes(full, noise)
    print("V5/V6/V7 common-MC=1000 audit complete; no merge or deletion was performed.")


if __name__ == "__main__":
    main()
