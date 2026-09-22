#!/usr/bin/env python3
"""Preregistered analysis of all four ImageNet R5 confirmatory methods."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_imagenet_confirmatory_r5_common as common


PROBABILITY_METRICS = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "label_coverage_k5_probability",
    "exact_set_k5_probability",
    "exact_set_k2_probability",
    "mean_logits_top1_accuracy",
    "mean_logits_exact_set_k5",
    "mean_logits_exact_set_k2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def validate_method_artifacts(
    protocol: Mapping[str, Any], method: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    directory = common.method_reeval_dir(Path(protocol["paths"]["output_root"]), method)
    rows_path = directory / "candidate_reeval_per_example.csv"
    summaries_path = directory / "candidate_reeval_summary.csv"
    method_path = directory / "method_summary.json"
    missing = [
        str(path) for path in (rows_path, summaries_path, method_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Incomplete ImageNet R5 method reevaluation:\n  " + "\n  ".join(missing)
        )
    rows = common.read_csv(rows_path)
    summaries = common.read_csv(summaries_path)
    method_summary = common.read_json(method_path)
    claimed = str(method_summary.get("method_summary_hash", ""))
    base = dict(method_summary)
    base.pop("method_summary_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError(f"ImageNet R5 method-summary hash mismatch: {method}")
    expected_header = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "status": "PASS",
        "method": method,
        "attack": common.METHOD_ATTACK[method],
        "search_seeds": list(common.SEARCH_SEEDS),
        "examples": common.CONFIRMATORY_EXAMPLES,
        "per_example_rows": common.CONFIRMATORY_EXAMPLES * len(common.SEARCH_SEEDS),
        "eval_draws": (
            common.REEVAL_DRAWS if common.method_spec(method).stochastic else 1
        ),
        "reeval_seed": common.REEVAL_SEED,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
        "paper_claim_authorized_by_method_stage": False,
    }
    for key, expected in expected_header.items():
        if method_summary.get(key) != expected:
            raise AssertionError(f"ImageNet R5 method summary changed: {method}/{key}")
    if method_summary.get("candidate_reeval_per_example_sha256") != common.file_sha256(rows_path):
        raise AssertionError(f"ImageNet R5 per-example hash mismatch: {method}")
    if method_summary.get("candidate_reeval_summary_sha256") != common.file_sha256(summaries_path):
        raise AssertionError(f"ImageNet R5 summary CSV hash mismatch: {method}")
    if len(rows) != 3000 or len(summaries) != 3:
        raise AssertionError(f"Unexpected ImageNet R5 table dimensions: {method}")

    expected_draws = common.REEVAL_DRAWS if common.method_spec(method).stochastic else 1
    groups: dict[int, list[dict[str, str]]] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if row.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError("ImageNet R5 row protocol ID changed")
        if row.get("protocol_hash") != protocol["protocol_hash"]:
            raise AssertionError("ImageNet R5 row protocol hash changed")
        if row.get("method") != method or row.get("attack") != common.METHOD_ATTACK[method]:
            raise AssertionError("ImageNet R5 row method/attack changed")
        seed = int(row["search_seed"])
        position = int(row["confirmatory_position"])
        if seed not in common.SEARCH_SEEDS or (seed, position) in seen:
            raise AssertionError("ImageNet R5 row duplication or unknown seed")
        seen.add((seed, position))
        groups.setdefault(seed, []).append(row)
        if int(row["eval_draws"]) != expected_draws:
            raise AssertionError("ImageNet R5 row MC draw count changed")
        if int(row["reeval_seed"]) != common.REEVAL_SEED:
            raise AssertionError("ImageNet R5 row reevaluation seed changed")
        if int(row["api_budget"]) != common.API_BUDGET:
            raise AssertionError("ImageNet R5 row API budget changed")
        for metric in PROBABILITY_METRICS:
            common.assert_probability(float(row[f"attacked_{metric}"]), f"attacked_{metric}")
            common.assert_probability(float(row[f"clean_{metric}"]), f"clean_{metric}")
        for metric in ("attacked_mean_boundary_margin_k5", "clean_mean_boundary_margin_k5"):
            if not np.isfinite(float(row[metric])):
                raise AssertionError(f"Non-finite ImageNet R5 margin: {metric}")
    if set(groups) != set(common.SEARCH_SEEDS):
        raise AssertionError(f"ImageNet R5 seed coverage changed: {method}")
    expected_positions = list(range(common.CONFIRMATORY_EXAMPLES))
    for seed, group in groups.items():
        if len(group) != common.CONFIRMATORY_EXAMPLES:
            raise AssertionError(f"ImageNet R5 seed row count changed: {method}/{seed}")
        if [int(row["confirmatory_position"]) for row in group] != expected_positions:
            raise AssertionError(f"ImageNet R5 seed order changed: {method}/{seed}")

    first = groups[common.SEARCH_SEEDS[0]]
    for seed in common.SEARCH_SEEDS[1:]:
        other = groups[seed]
        for anchor, row in zip(first, other):
            for key in ("relative_path", "label", "reference_top1", "reference_set_k5"):
                if row[key] != anchor[key]:
                    raise AssertionError(f"ImageNet R5 anchors differ across seeds: {method}/{key}")
            for metric in (*PROBABILITY_METRICS, "mean_boundary_margin_k5"):
                if float(row[f"clean_{metric}"]) != float(anchor[f"clean_{metric}"]):
                    raise AssertionError(f"ImageNet R5 clean values differ across seeds: {method}/{metric}")

    summary_map = {int(row["search_seed"]): row for row in summaries}
    if set(summary_map) != set(common.SEARCH_SEEDS):
        raise AssertionError(f"ImageNet R5 candidate-summary seed coverage changed: {method}")
    for seed, group in groups.items():
        summary = summary_map[seed]
        if summary["attack"] != common.METHOD_ATTACK[method]:
            raise AssertionError("ImageNet R5 candidate-summary attack changed")
        checks = {
            "attacked_exact_set_k5_probability_percent": 100.0
            * float(np.mean([float(row["attacked_exact_set_k5_probability"]) for row in group])),
            "clean_exact_set_k5_probability_percent": 100.0
            * float(np.mean([float(row["clean_exact_set_k5_probability"]) for row in group])),
            "attacked_top1_accuracy_probability_percent": 100.0
            * float(np.mean([float(row["attacked_top1_accuracy_probability"]) for row in group])),
            "clean_top1_accuracy_probability_percent": 100.0
            * float(np.mean([float(row["clean_top1_accuracy_probability"]) for row in group])),
        }
        for key, expected in checks.items():
            if abs(float(summary[key]) - expected) > 1e-9:
                raise AssertionError(f"ImageNet R5 aggregate mismatch: {method}/{seed}/{key}")
    return rows, summaries


def validate_cross_method_alignment(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    reference: list[tuple[int, str, int, int, str]] | None = None
    for method in common.METHODS:
        subset = [
            row for row in rows
            if row["method"] == method and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
        ]
        subset.sort(key=lambda row: int(row["confirmatory_position"]))
        signature = [
            (
                int(row["confirmatory_position"]),
                str(row["relative_path"]),
                int(row["label"]),
                int(row["reference_top1"]),
                str(row["reference_set_k5"]),
            )
            for row in subset
        ]
        if reference is None:
            reference = signature
        elif signature != reference:
            raise AssertionError("ImageNet R5 methods do not share samples, labels, and references")
    if reference is None or len(reference) != common.CONFIRMATORY_EXAMPLES:
        raise AssertionError("ImageNet R5 cross-method alignment is incomplete")
    common.validate_confirmatory_labels([item[2] for item in reference])
    return [item[0] for item in reference]


def build_method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        seed_exact5: list[float] = []
        seed_exact2: list[float] = []
        seed_top1: list[float] = []
        for seed in common.SEARCH_SEEDS:
            group = [
                row for row in rows
                if row["method"] == method and int(row["search_seed"]) == seed
            ]
            if len(group) != common.CONFIRMATORY_EXAMPLES:
                raise AssertionError(f"Missing ImageNet R5 group: {method}/{seed}")
            seed_exact5.append(100.0 * float(np.mean([
                float(row["attacked_exact_set_k5_probability"]) for row in group
            ])))
            seed_exact2.append(100.0 * float(np.mean([
                float(row["attacked_exact_set_k2_probability"]) for row in group
            ])))
            seed_top1.append(100.0 * float(np.mean([
                float(row["attacked_top1_accuracy_probability"]) for row in group
            ])))
        _, attacked_exact5 = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_k5_probability"
        )
        _, attacked_exact2 = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_k2_probability"
        )
        _, attacked_top1 = common.aligned_values(
            rows, method=method, metric="attacked_top1_accuracy_probability"
        )
        _, attacked_coverage5 = common.aligned_values(
            rows, method=method, metric="attacked_label_coverage_k5_probability"
        )
        _, clean_exact5 = common.clean_values(
            rows, method=method, metric="clean_exact_set_k5_probability"
        )
        _, clean_exact2 = common.clean_values(
            rows, method=method, metric="clean_exact_set_k2_probability"
        )
        _, clean_top1 = common.clean_values(
            rows, method=method, metric="clean_top1_accuracy_probability"
        )
        _, clean_coverage5 = common.clean_values(
            rows, method=method, metric="clean_label_coverage_k5_probability"
        )
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "method": method,
                "attack": common.METHOD_ATTACK[method],
                "examples": common.CONFIRMATORY_EXAMPLES,
                "search_seeds": "|".join(str(seed) for seed in common.SEARCH_SEEDS),
                "attacked_exact_set_k5_probability_percent": 100.0 * float(attacked_exact5.mean()),
                "attacked_exact_set_k5_seed_sd_pp": float(statistics.stdev(seed_exact5)),
                "attacked_exact_set_k5_seed_values_percent": "|".join(
                    f"{value:.6f}" for value in seed_exact5
                ),
                "clean_exact_set_k5_probability_percent": 100.0 * float(clean_exact5.mean()),
                "clean_to_attacked_exact_set_k5_delta_pp": 100.0
                * float((attacked_exact5 - clean_exact5).mean()),
                "attacked_exact_set_k2_probability_percent": 100.0 * float(attacked_exact2.mean()),
                "attacked_exact_set_k2_seed_sd_pp": float(statistics.stdev(seed_exact2)),
                "attacked_exact_set_k2_seed_values_percent": "|".join(
                    f"{value:.6f}" for value in seed_exact2
                ),
                "clean_exact_set_k2_probability_percent": 100.0 * float(clean_exact2.mean()),
                "attacked_top1_accuracy_probability_percent": 100.0 * float(attacked_top1.mean()),
                "attacked_top1_seed_sd_pp": float(statistics.stdev(seed_top1)),
                "attacked_top1_seed_values_percent": "|".join(
                    f"{value:.6f}" for value in seed_top1
                ),
                "clean_top1_accuracy_probability_percent": 100.0 * float(clean_top1.mean()),
                "attacked_label_coverage_k5_probability_percent": 100.0
                * float(attacked_coverage5.mean()),
                "clean_label_coverage_k5_probability_percent": 100.0
                * float(clean_coverage5.mean()),
                "attack_family_frozen_on_disjoint_pilot": True,
                "seed_selection_performed": False,
            }
        )
    return output


def build_classwise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        positions, attacked = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_k5_probability"
        )
        clean_positions, clean = common.clean_values(
            rows, method=method, metric="clean_exact_set_k5_probability"
        )
        if positions != clean_positions:
            raise AssertionError("ImageNet R5 classwise clean/attack alignment changed")
        first_seed = [
            row for row in rows
            if row["method"] == method and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
        ]
        labels = {int(row["confirmatory_position"]): int(row["label"]) for row in first_seed}
        paths = {int(row["confirmatory_position"]): str(row["relative_path"]) for row in first_seed}
        inverse = {label: position for position, label in labels.items()}
        if set(inverse) != set(range(1000)):
            raise AssertionError("ImageNet R5 classwise labels are incomplete")
        for class_id in range(1000):
            position = inverse[class_id]
            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "method": method,
                    "attack": common.METHOD_ATTACK[method],
                    "class_id": class_id,
                    "confirmatory_position": position,
                    "relative_path": paths[position],
                    "examples": 1,
                    "clean_exact_set_k5_probability_percent": 100.0 * float(clean[position]),
                    "attacked_exact_set_k5_probability_percent": 100.0 * float(attacked[position]),
                    "clean_to_attacked_exact_set_k5_delta_pp": 100.0
                    * float(attacked[position] - clean[position]),
                }
            )
    return output


def build_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    attacked_exact5: dict[str, np.ndarray] = {}
    attacked_exact2: dict[str, np.ndarray] = {}
    attacked_top1: dict[str, np.ndarray] = {}
    clean_exact5: dict[str, np.ndarray] = {}
    common_positions: list[int] | None = None
    for method in common.METHODS:
        positions, attacked_exact5[method] = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_k5_probability"
        )
        p2, attacked_exact2[method] = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_k2_probability"
        )
        p1, attacked_top1[method] = common.aligned_values(
            rows, method=method, metric="attacked_top1_accuracy_probability"
        )
        pc, clean_exact5[method] = common.clean_values(
            rows, method=method, metric="clean_exact_set_k5_probability"
        )
        if not (positions == p2 == p1 == pc):
            raise AssertionError("ImageNet R5 comparison positions are not aligned")
        if common_positions is None:
            common_positions = positions
        elif positions != common_positions:
            raise AssertionError("ImageNet R5 comparison methods are not aligned")

    comparisons: list[tuple[str, str, np.ndarray]] = [
        (
            common.PRIMARY_COMPARISON,
            "primary",
            attacked_exact5["full"] - attacked_exact5["robust"],
        ),
        (
            "noise_only_minus_robust_attacked_exact_set_k5_probability",
            "secondary",
            attacked_exact5["noise_only"] - attacked_exact5["robust"],
        ),
        (
            "full_sigma0_minus_robust_attacked_exact_set_k5_probability",
            "secondary",
            attacked_exact5["full_sigma0"] - attacked_exact5["robust"],
        ),
        (
            "full_minus_noise_only_attacked_exact_set_k5_probability",
            "secondary",
            attacked_exact5["full"] - attacked_exact5["noise_only"],
        ),
        (
            "full_minus_full_sigma0_attacked_exact_set_k5_probability",
            "secondary",
            attacked_exact5["full"] - attacked_exact5["full_sigma0"],
        ),
    ]
    drops = {
        method: attacked_exact5[method] - clean_exact5[method] for method in common.METHODS
    }
    for method in common.METHODS:
        comparisons.append(
            (f"{method}_clean_to_attacked_exact_set_k5_probability", "diagnostic", drops[method])
        )
    comparisons.extend(
        [
            (
                "full_vs_robust_difference_in_exact_set_k5_drops",
                "secondary",
                drops["full"] - drops["robust"],
            ),
            (
                "noise_only_vs_robust_difference_in_exact_set_k5_drops",
                "secondary",
                drops["noise_only"] - drops["robust"],
            ),
            (
                "full_sigma0_vs_robust_difference_in_exact_set_k5_drops",
                "secondary",
                drops["full_sigma0"] - drops["robust"],
            ),
            (
                "full_vs_noise_only_difference_in_exact_set_k5_drops",
                "secondary",
                drops["full"] - drops["noise_only"],
            ),
            (
                "full_minus_robust_attacked_exact_set_k2_probability",
                "secondary_k2",
                attacked_exact2["full"] - attacked_exact2["robust"],
            ),
            (
                "noise_only_minus_robust_attacked_exact_set_k2_probability",
                "secondary_k2",
                attacked_exact2["noise_only"] - attacked_exact2["robust"],
            ),
            (
                "full_minus_robust_attacked_top1_accuracy_probability",
                "secondary_top1",
                attacked_top1["full"] - attacked_top1["robust"],
            ),
            (
                "noise_only_minus_robust_attacked_top1_accuracy_probability",
                "secondary_top1",
                attacked_top1["noise_only"] - attacked_top1["robust"],
            ),
        ]
    )
    output: list[dict[str, Any]] = []
    for name, family, delta in comparisons:
        result = common.paired_bootstrap(delta)
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "comparison": name,
                "family": family,
                "estimand": "confirmatory_fixed_attack_three_seed_mean_per_image",
                **result,
                "preregistered": True,
                "paper_claim_authorized": (
                    name == common.PRIMARY_COMPARISON and result["status"] == "positive"
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir != output_root / "confirmatory_analysis":
        raise ValueError("ImageNet R5 analysis path differs from the frozen protocol")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    all_candidate_summaries: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, summaries = validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
        all_candidate_summaries.extend(summaries)
    positions = validate_cross_method_alignment(all_rows)
    if positions != list(range(common.CONFIRMATORY_EXAMPLES)):
        raise AssertionError("ImageNet R5 final positions are incomplete")

    method_summary = build_method_summary(all_rows)
    comparisons = build_comparisons(all_rows)
    classwise = build_classwise(all_rows)
    primary = next(
        row for row in comparisons if row["comparison"] == common.PRIMARY_COMPARISON
    )
    primary_pass = primary["status"] == "positive"

    per_example_path = output_dir / "confirmatory_per_example.csv"
    candidate_summary_path = output_dir / "confirmatory_candidate_summary.csv"
    method_summary_path = output_dir / "confirmatory_method_summary.csv"
    comparison_path = output_dir / "confirmatory_paired_bootstrap.csv"
    classwise_path = output_dir / "confirmatory_classwise.csv"
    common.atomic_csv(per_example_path, all_rows)
    common.atomic_csv(candidate_summary_path, all_candidate_summaries)
    common.atomic_csv(method_summary_path, method_summary)
    common.atomic_csv(comparison_path, comparisons)
    common.atomic_csv(classwise_path, classwise)

    handoff: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "authorized": primary_pass,
        "authorization_scope": (
            protocol["analysis_plan"]["primary_claim_scope"] if primary_pass else None
        ),
        "primary_comparison": primary,
        "secondary_results_are_not_promoted_to_primary": True,
        "no_attack_family_selection_on_confirmatory_data": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
        "claim_text_if_authorized": (
            "On the preregistered disjoint 1,000-class ImageNet-1k holdout, the "
            "frozen full defense achieved higher attacked Exact Top-5 set probability "
            "than the frozen robust ResNet-50 anchor under method-specific attacks "
            "selected only on the disjoint pilot."
            if primary_pass
            else None
        ),
    }
    handoff["handoff_hash"] = common.object_sha256(handoff)
    handoff_path = output_dir / "paper_claim_handoff.json"
    common.atomic_json(handoff_path, handoff)

    final: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "examples": common.CONFIRMATORY_EXAMPLES,
        "one_image_per_class": True,
        "methods": list(common.METHODS),
        "method_attack": dict(common.METHOD_ATTACK),
        "search_seeds": list(common.SEARCH_SEEDS),
        "api_budget": common.API_BUDGET,
        "epsilon_px": common.EPSILON_PX,
        "common_mc_draws_stochastic": common.REEVAL_DRAWS,
        "method_summary": method_summary,
        "primary_result": primary,
        "paper_claim_authorized": primary_pass,
        "paper_claim_handoff_hash": handoff["handoff_hash"],
        "confirmatory_attack_selection_performed": False,
        "confirmatory_seed_selection_performed": False,
        "confirmatory_per_example_selection_performed": False,
        "defense_retuning_performed": False,
        "files": {
            "confirmatory_per_example.csv": common.file_sha256(per_example_path),
            "confirmatory_candidate_summary.csv": common.file_sha256(candidate_summary_path),
            "confirmatory_method_summary.csv": common.file_sha256(method_summary_path),
            "confirmatory_paired_bootstrap.csv": common.file_sha256(comparison_path),
            "confirmatory_classwise.csv": common.file_sha256(classwise_path),
            "paper_claim_handoff.json": common.file_sha256(handoff_path),
        },
    }
    final["final_summary_hash"] = common.object_sha256(final)
    common.atomic_json(output_dir / "final_summary.json", final)

    print("\nImageNet R5 preregistered confirmatory results", flush=True)
    for row in method_summary:
        print(
            f"  {row['method']:>11}: attack={row['attack']} "
            f"clean Exact5={float(row['clean_exact_set_k5_probability_percent']):.3f}% "
            f"attacked Exact5={float(row['attacked_exact_set_k5_probability_percent']):.3f}% "
            f"sd={float(row['attacked_exact_set_k5_seed_sd_pp']):.3f} pp",
            flush=True,
        )
    print(
        f"PRIMARY Full-Robust={float(primary['difference_pp']):+.3f} pp "
        f"[{float(primary['ci95_low_pp']):+.3f}, "
        f"{float(primary['ci95_high_pp']):+.3f}] -> {primary['status']}",
        flush=True,
    )
    print(f"PAPER_CLAIM_AUTHORIZED={'YES' if primary_pass else 'NO'}", flush=True)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
