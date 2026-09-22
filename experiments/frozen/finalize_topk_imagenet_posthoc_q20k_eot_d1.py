#!/usr/bin/env python3
"""Finalize the D1 post-hoc Q=20k stress diagnostic without changing R5."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_imagenet_posthoc_q20k_eot_d1_common as common
from reevaluate_topk_imagenet_posthoc_q20k_eot_d1 import METRICS


PROBABILITY_METRICS = tuple(
    metric for metric in METRICS if metric != "mean_boundary_margin_k5"
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


def assert_probability(value: float, label: str) -> None:
    if not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
        raise AssertionError(f"Probability outside [0,1]: {label}={value}")


def validate_method_artifacts(
    protocol: Mapping[str, Any], method: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    directory = common.method_reeval_dir(
        Path(protocol["paths"]["output_root"]), method
    )
    rows_path = directory / "candidate_reeval_per_example.csv"
    summaries_path = directory / "candidate_reeval_summary.csv"
    method_path = directory / "method_summary.json"
    missing = [
        str(path)
        for path in (rows_path, summaries_path, method_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Incomplete D1 reevaluation:\n  " + "\n  ".join(missing))
    rows = common.read_csv(rows_path)
    summaries = common.read_csv(summaries_path)
    method_summary = common.read_json(method_path)
    claimed = str(method_summary.get("method_summary_hash", ""))
    base = dict(method_summary)
    base.pop("method_summary_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError(f"D1 method-summary hash mismatch: {method}")
    expected_draws = (
        common.REEVAL_DRAWS if common.method_spec(method).stochastic else 1
    )
    expected_header = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "status": "PASS",
        "method": method,
        "attack": common.METHOD_ATTACK[method],
        "search_seeds": list(common.SEARCH_SEEDS),
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "per_example_rows": common.DIAGNOSTIC_EXAMPLES
        * len(common.SEARCH_SEEDS),
        "eval_draws": expected_draws,
        "reeval_seed": common.REEVAL_SEED,
        "common_random_numbers": True,
        "post_hoc": True,
        "paper_claim_authorized": False,
        "parent_r5_modified": False,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
    }
    for key, expected in expected_header.items():
        if method_summary.get(key) != expected:
            raise AssertionError(f"D1 method summary changed: {method}/{key}")
    if method_summary.get("candidate_reeval_per_example_sha256") != (
        common.file_sha256(rows_path)
    ):
        raise AssertionError(f"D1 per-example hash mismatch: {method}")
    if method_summary.get("candidate_reeval_summary_sha256") != (
        common.file_sha256(summaries_path)
    ):
        raise AssertionError(f"D1 summary-table hash mismatch: {method}")
    if len(rows) != common.DIAGNOSTIC_EXAMPLES * len(common.SEARCH_SEEDS):
        raise AssertionError(f"Unexpected D1 per-example row count: {method}")
    if len(summaries) != len(common.SEARCH_SEEDS):
        raise AssertionError(f"Unexpected D1 summary row count: {method}")

    groups: dict[int, list[dict[str, str]]] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if (
            row.get("protocol_id") != common.PROTOCOL_ID
            or row.get("protocol_hash") != protocol["protocol_hash"]
            or row.get("method") != method
            or row.get("attack") != common.METHOD_ATTACK[method]
        ):
            raise AssertionError("D1 per-example identity changed")
        seed = int(row["search_seed"])
        position = int(row["diagnostic_position"])
        if seed not in common.SEARCH_SEEDS or (seed, position) in seen:
            raise AssertionError("D1 duplicate row or unknown seed")
        seen.add((seed, position))
        groups.setdefault(seed, []).append(row)
        if (
            int(row["eval_draws"]) != expected_draws
            or int(row["reeval_seed"]) != common.REEVAL_SEED
            or int(row["api_budget"]) != common.API_BUDGET
            or float(row["epsilon_px"]) != common.EPSILON_PX
            or int(row["search_eot"])
            != common.ADAPTIVE_CONFIG_BY_METHOD[method]["search_eot"]
            or not as_bool(row["post_hoc"])
            or as_bool(row["paper_claim_authorized"])
        ):
            raise AssertionError("D1 frozen evaluation fields changed")
        for metric in PROBABILITY_METRICS:
            assert_probability(float(row[f"attacked_{metric}"]), metric)
            assert_probability(float(row[f"clean_{metric}"]), metric)
        for prefix in ("attacked", "clean"):
            value = float(row[f"{prefix}_mean_boundary_margin_k5"])
            if not math.isfinite(value):
                raise AssertionError("D1 boundary margin is non-finite")
    if set(groups) != set(common.SEARCH_SEEDS):
        raise AssertionError(f"D1 seed coverage changed: {method}")
    for seed in common.SEARCH_SEEDS:
        group = groups[seed]
        if [int(row["diagnostic_position"]) for row in group] != list(
            range(common.DIAGNOSTIC_EXAMPLES)
        ):
            raise AssertionError(f"D1 position order changed: {method}/{seed}")

    first = groups[common.SEARCH_SEEDS[0]]
    for seed in common.SEARCH_SEEDS[1:]:
        for anchor, row in zip(first, groups[seed]):
            for key in (
                "source_confirmatory_position",
                "relative_path",
                "label",
                "reference_top1",
                "reference_set_k5",
            ):
                if row[key] != anchor[key]:
                    raise AssertionError(
                        f"D1 anchors differ across seeds: {method}/{key}"
                    )
            for metric in METRICS:
                if float(row[f"clean_{metric}"]) != float(
                    anchor[f"clean_{metric}"]
                ):
                    raise AssertionError(
                        f"D1 clean values differ across seeds: {method}/{metric}"
                    )

    summary_map = {int(row["search_seed"]): row for row in summaries}
    if set(summary_map) != set(common.SEARCH_SEEDS):
        raise AssertionError(f"D1 summary seed coverage changed: {method}")
    for seed, group in groups.items():
        summary = summary_map[seed]
        expected_exact5 = 100.0 * float(
            np.mean(
                [float(row["attacked_exact_set_k5_probability"]) for row in group]
            )
        )
        if abs(
            float(summary["attacked_exact_set_k5_probability_percent"])
            - expected_exact5
        ) > 1e-9:
            raise AssertionError(f"D1 aggregate mismatch: {method}/{seed}")
        if not as_bool(summary["post_hoc"]) or as_bool(
            summary["paper_claim_authorized"]
        ):
            raise AssertionError("D1 candidate summary changed its diagnostic status")
    return rows, summaries


def validate_cross_method_alignment(rows: Sequence[Mapping[str, Any]]) -> None:
    reference: list[tuple[int, int, str, int, int, str]] | None = None
    for method in common.METHODS:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
        ]
        subset.sort(key=lambda row: int(row["diagnostic_position"]))
        signature = [
            (
                int(row["diagnostic_position"]),
                int(row["source_confirmatory_position"]),
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
            raise AssertionError("D1 methods do not share samples and references")
    if reference is None or len(reference) != common.DIAGNOSTIC_EXAMPLES:
        raise AssertionError("D1 cross-method alignment is incomplete")


def aligned_seed_mean(
    rows: Sequence[Mapping[str, Any]], method: str, field: str
) -> np.ndarray:
    matrix: list[list[float]] = []
    for seed in common.SEARCH_SEEDS:
        selected = [
            row
            for row in rows
            if row["method"] == method and int(row["search_seed"]) == seed
        ]
        selected.sort(key=lambda row: int(row["diagnostic_position"]))
        if [int(row["diagnostic_position"]) for row in selected] != list(
            range(common.DIAGNOSTIC_EXAMPLES)
        ):
            raise AssertionError(f"Incomplete D1 aligned group: {method}/{seed}")
        matrix.append([float(row[field]) for row in selected])
    return np.asarray(matrix, dtype=np.float64).mean(0)


def clean_values(
    rows: Sequence[Mapping[str, Any]], method: str, field: str
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["method"] == method
        and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
    ]
    selected.sort(key=lambda row: int(row["diagnostic_position"]))
    if len(selected) != common.DIAGNOSTIC_EXAMPLES:
        raise AssertionError(f"Incomplete D1 clean group: {method}")
    return np.asarray([float(row[field]) for row in selected], dtype=np.float64)


def parent_seed_mean(
    parent_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    field: str,
    subset: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    parent_seeds = sorted(
        {
            int(row["search_seed"])
            for row in parent_rows
            if row.get("method") == method
        }
    )
    if len(parent_seeds) != 3:
        raise AssertionError(f"R5 parent seed coverage changed: {method}")
    source_positions = [int(row["source_confirmatory_position"]) for row in subset]
    source_set = set(source_positions)
    matrix: list[list[float]] = []
    for seed in parent_seeds:
        selected = {
            int(row["confirmatory_position"]): row
            for row in parent_rows
            if row.get("method") == method
            and int(row["search_seed"]) == seed
            and int(row["confirmatory_position"]) in source_set
        }
        if set(selected) != source_set:
            raise AssertionError(f"R5 parent subset is incomplete: {method}/{seed}")
        ordered: list[float] = []
        for subset_row in subset:
            source = int(subset_row["source_confirmatory_position"])
            row = selected[source]
            if (
                row["relative_path"] != subset_row["relative_path"]
                or int(row["label"]) != int(subset_row["target"])
            ):
                raise AssertionError("R5/D1 subset anchor mismatch")
            ordered.append(float(row[field]))
        matrix.append(ordered)
    return np.asarray(matrix, dtype=np.float64).mean(0)


def build_method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        seed_exact5: list[float] = []
        seed_exact2: list[float] = []
        seed_top1: list[float] = []
        for seed in common.SEARCH_SEEDS:
            group = [
                row
                for row in rows
                if row["method"] == method and int(row["search_seed"]) == seed
            ]
            seed_exact5.append(
                100.0
                * float(
                    np.mean(
                        [
                            float(row["attacked_exact_set_k5_probability"])
                            for row in group
                        ]
                    )
                )
            )
            seed_exact2.append(
                100.0
                * float(
                    np.mean(
                        [
                            float(row["attacked_exact_set_k2_probability"])
                            for row in group
                        ]
                    )
                )
            )
            seed_top1.append(
                100.0
                * float(
                    np.mean(
                        [
                            float(row["attacked_top1_accuracy_probability"])
                            for row in group
                        ]
                    )
                )
            )
        attacked5 = aligned_seed_mean(
            rows, method, "attacked_exact_set_k5_probability"
        )
        attacked2 = aligned_seed_mean(
            rows, method, "attacked_exact_set_k2_probability"
        )
        attacked1 = aligned_seed_mean(
            rows, method, "attacked_top1_accuracy_probability"
        )
        clean5 = clean_values(rows, method, "clean_exact_set_k5_probability")
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "method": method,
                "attack": common.METHOD_ATTACK[method],
                "examples": common.DIAGNOSTIC_EXAMPLES,
                "search_seeds": "|".join(str(seed) for seed in common.SEARCH_SEEDS),
                "search_eot": common.ADAPTIVE_CONFIG_BY_METHOD[method]["search_eot"],
                "api_budget": common.API_BUDGET,
                "attacked_exact_set_k5_probability_percent": 100.0
                * float(attacked5.mean()),
                "attacked_exact_set_k5_seed_sd_pp": statistics.stdev(seed_exact5),
                "attacked_exact_set_k5_seed_values_percent": "|".join(
                    f"{value:.6f}" for value in seed_exact5
                ),
                "clean_exact_set_k5_probability_percent": 100.0
                * float(clean5.mean()),
                "clean_to_attacked_exact_set_k5_delta_pp": 100.0
                * float((attacked5 - clean5).mean()),
                "attacked_exact_set_k2_probability_percent": 100.0
                * float(attacked2.mean()),
                "attacked_exact_set_k2_seed_sd_pp": statistics.stdev(seed_exact2),
                "attacked_top1_accuracy_probability_percent": 100.0
                * float(attacked1.mean()),
                "attacked_top1_seed_sd_pp": statistics.stdev(seed_top1),
                "post_hoc": True,
                "paper_claim_authorized": False,
                "seed_selection_performed": False,
            }
        )
    return output


def build_budget_rows(
    rows: Sequence[Mapping[str, Any]],
    parent_rows: Sequence[Mapping[str, Any]],
    subset: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    for method in common.METHODS:
        values[f"{method}_q20_exact5"] = aligned_seed_mean(
            rows, method, "attacked_exact_set_k5_probability"
        )
        values[f"{method}_q5_exact5"] = parent_seed_mean(
            parent_rows,
            method=method,
            field="attacked_exact_set_k5_probability",
            subset=subset,
        )
        values[f"{method}_q20_exact2"] = aligned_seed_mean(
            rows, method, "attacked_exact_set_k2_probability"
        )
        values[f"{method}_q5_exact2"] = parent_seed_mean(
            parent_rows,
            method=method,
            field="attacked_exact_set_k2_probability",
            subset=subset,
        )
        values[f"{method}_q20_top1"] = aligned_seed_mean(
            rows, method, "attacked_top1_accuracy_probability"
        )
        values[f"{method}_q5_top1"] = parent_seed_mean(
            parent_rows,
            method=method,
            field="attacked_top1_accuracy_probability",
            subset=subset,
        )
    output: list[dict[str, Any]] = []
    for position, subset_row in enumerate(subset):
        row: dict[str, Any] = {
            "protocol_id": common.PROTOCOL_ID,
            "diagnostic_position": position,
            "source_confirmatory_position": subset_row[
                "source_confirmatory_position"
            ],
            "relative_path": subset_row["relative_path"],
            "label": subset_row["target"],
            "post_hoc": True,
            "paper_claim_authorized": False,
        }
        for method in common.METHODS:
            for metric in ("exact5", "exact2", "top1"):
                q5 = values[f"{method}_q5_{metric}"][position]
                q20 = values[f"{method}_q20_{metric}"][position]
                row[f"{method}_q5_{metric}_probability"] = float(q5)
                row[f"{method}_q20_{metric}_probability"] = float(q20)
                row[f"{method}_q20_minus_q5_{metric}_probability"] = float(
                    q20 - q5
                )
        output.append(row)
    return output, values


def comparison_row(
    name: str, family: str, delta: np.ndarray
) -> dict[str, Any]:
    return {
        "protocol_id": common.PROTOCOL_ID,
        "comparison": name,
        "family": family,
        "estimand": "fixed_100_image_three_seed_mean_per_image",
        **common.paired_bootstrap(delta),
        "post_hoc": True,
        "preregistered_r5": False,
        "paper_claim_authorized": False,
    }


def build_comparisons(values: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    full20 = values["full_q20_exact5"]
    robust20 = values["robust_q20_exact5"]
    full5 = values["full_q5_exact5"]
    robust5 = values["robust_q5_exact5"]
    return [
        comparison_row(
            "full_minus_robust_q20_attacked_exact_set_k5_probability",
            "primary_diagnostic",
            full20 - robust20,
        ),
        comparison_row(
            "full_q20_eot5_minus_r5_q5_eot1_attacked_exact_set_k5_probability",
            "query_eot_stress",
            full20 - full5,
        ),
        comparison_row(
            "robust_q20_eot1_minus_r5_q5_eot1_attacked_exact_set_k5_probability",
            "query_eot_stress",
            robust20 - robust5,
        ),
        comparison_row(
            "full_minus_robust_q20_eotstress_vs_r5_q5_difference_in_differences_exact_set_k5",
            "query_eot_stress",
            (full20 - full5) - (robust20 - robust5),
        ),
        comparison_row(
            "full_minus_robust_q20_attacked_exact_set_k2_probability",
            "secondary_k2",
            values["full_q20_exact2"] - values["robust_q20_exact2"],
        ),
        comparison_row(
            "full_minus_robust_q20_attacked_top1_accuracy_probability",
            "secondary_top1",
            values["full_q20_top1"] - values["robust_q20_top1"],
        ),
    ]


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_json.expanduser().resolve()
    protocol = common.read_json(protocol_path)
    protocol_hash = common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    if protocol_path != output_root / "diagnostic_protocol.json":
        raise AssertionError("D1 protocol path is not canonical")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir != (output_root / "diagnostic_analysis").resolve():
        raise ValueError("D1 analysis path differs from the frozen protocol")
    output_dir.mkdir(parents=True, exist_ok=True)

    parent = common.load_parent_r5(Path(protocol["paths"]["r5_root"]))
    parent_rows = common.read_csv(parent["paths"]["per_example"])
    all_rows: list[dict[str, str]] = []
    all_candidate_summaries: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, summaries = validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
        all_candidate_summaries.extend(summaries)
    validate_cross_method_alignment(all_rows)

    method_summary = build_method_summary(all_rows)
    budget_rows, values = build_budget_rows(all_rows, parent_rows, parent["subset"])
    comparisons = build_comparisons(values)
    primary = next(
        row for row in comparisons if row["family"] == "primary_diagnostic"
    )

    per_example_path = output_dir / "diagnostic_per_example.csv"
    candidate_path = output_dir / "diagnostic_candidate_summary.csv"
    method_path = output_dir / "diagnostic_method_summary.csv"
    budget_path = (
        output_dir / "diagnostic_q20k_eot_stress_comparison_per_example.csv"
    )
    comparison_path = output_dir / "diagnostic_paired_bootstrap.csv"
    common.atomic_csv(per_example_path, all_rows)
    common.atomic_csv(candidate_path, all_candidate_summaries)
    common.atomic_csv(method_path, method_summary)
    common.atomic_csv(budget_path, budget_rows)
    common.atomic_csv(comparison_path, comparisons)

    files = {
        path.name: common.file_sha256(path)
        for path in (
            per_example_path,
            candidate_path,
            method_path,
            budget_path,
            comparison_path,
        )
    }
    final: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "scope": "post-hoc query-budget/EOT masking stress diagnostic",
        "methods": list(common.METHODS),
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "search_seeds": list(common.SEARCH_SEEDS),
        "api_budget": common.API_BUDGET,
        "epsilon_px": common.EPSILON_PX,
        "reeval_draws_stochastic": common.REEVAL_DRAWS,
        "reeval_seed": common.REEVAL_SEED,
        "budget_and_eot_effects_separately_identifiable": False,
        "r5_vs_d1_reevaluation_common_random_numbers": False,
        "primary_diagnostic": primary,
        "parent_r5_protocol_hash": common.EXPECTED_R5_PROTOCOL_HASH,
        "parent_r5_final_summary_hash": common.EXPECTED_R5_FINAL_SUMMARY_HASH,
        "parent_r5_paper_claim_remains_authorized": True,
        "parent_r5_modified": False,
        "post_hoc": True,
        "paper_claim_authorized": False,
        "may_revise_parent_r5_claim": False,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "files": files,
    }
    final["final_summary_hash"] = common.object_sha256(final)
    common.atomic_json(output_dir / "final_summary.json", final)
    common.atomic_json(
        output_dir / "claim_boundary.json",
        {
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol_hash,
            "diagnostic_complete": True,
            "post_hoc": True,
            "paper_claim_authorized": False,
            "may_revise_parent_r5_claim": False,
            "parent_r5_claim_status": "unchanged_and_authorized_by_R5_only",
            "parent_r5_protocol_hash": common.EXPECTED_R5_PROTOCOL_HASH,
            "parent_r5_final_summary_hash": common.EXPECTED_R5_FINAL_SUMMARY_HASH,
        },
    )

    print("ImageNet ResNet-50 D1 post-hoc Q20k results", flush=True)
    for row in method_summary:
        print(
            f"  {row['method']:>6}: EOT={row['search_eot']} "
            f"Exact5={float(row['attacked_exact_set_k5_probability_percent']):.3f}% "
            f"sd={float(row['attacked_exact_set_k5_seed_sd_pp']):.3f} pp",
            flush=True,
        )
    print(
        "DIAGNOSTIC Full-Robust="
        f"{float(primary['difference_pp']):+.3f} pp "
        f"[{float(primary['ci95_low_pp']):+.3f}, "
        f"{float(primary['ci95_high_pp']):+.3f}] -> {primary['status']}",
        flush=True,
    )
    for row in comparisons:
        if row["family"] == "query_eot_stress":
            print(
                f"STRESS {row['comparison']}: "
                f"{float(row['difference_pp']):+.3f} pp "
                f"[{float(row['ci95_low_pp']):+.3f}, "
                f"{float(row['ci95_high_pp']):+.3f}] -> {row['status']}",
                flush=True,
            )
    print("PAPER_CLAIM_AUTHORIZED=NO (post-hoc diagnostic)", flush=True)
    print("R5_PAPER_CLAIM_STATUS=UNCHANGED", flush=True)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
