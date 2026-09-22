#!/usr/bin/env python3
"""Finalize the exploratory CIFAR-100 R6 Q=5k/20k stress test."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_cifar100_query_stress_r6_common as common


PROBABILITY_METRICS = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "exact_set_probability",
    "label_coverage_probability",
    "mean_logits_top1_accuracy",
    "mean_logits_exact_set",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize CIFAR-100 query stress R6")
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_method_artifacts(
    protocol: Mapping[str, Any], method: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    directory = common.method_reeval_dir(Path(protocol["output_root"]), method)
    rows_path = directory / "candidate_reeval_per_example.csv"
    summaries_path = directory / "candidate_reeval_summary.csv"
    final_path = directory / "method_summary.json"
    missing = [str(path) for path in (rows_path, summaries_path, final_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete R6 method reevaluation:\n  " + "\n  ".join(missing))
    rows = common.read_csv(rows_path)
    summaries = common.read_csv(summaries_path)
    final = common.read_json(final_path)
    claimed = str(final.get("method_summary_hash", ""))
    base = dict(final)
    base.pop("method_summary_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError(f"R6 method-summary hash mismatch: {method}")
    if final.get("protocol_hash") != protocol["protocol_hash"]:
        raise AssertionError(f"R6 method/protocol identity mismatch: {method}")
    if final.get("status") != "PASS" or final.get("method") != method:
        raise AssertionError(f"R6 method stage did not pass: {method}")
    if final.get("attack") != common.METHOD_ATTACK[method]:
        raise AssertionError(f"R6 method attack mismatch: {method}")
    if final.get("budgets") != list(common.BUDGETS):
        raise AssertionError(f"R6 method budgets changed: {method}")
    if final.get("search_seeds") != list(common.SEARCH_SEEDS):
        raise AssertionError(f"R6 method seeds changed: {method}")
    expected_rows = len(common.BUDGETS) * len(common.SEARCH_SEEDS) * common.STRESS_EXAMPLES
    if int(final.get("examples", -1)) != common.STRESS_EXAMPLES:
        raise AssertionError(f"R6 method example count changed: {method}")
    if int(final.get("per_example_rows", -1)) != expected_rows:
        raise AssertionError(f"R6 method row count changed: {method}")
    for key in (
        "no_attack_family_selection",
        "no_seed_selection",
        "no_per_example_budget_selection",
    ):
        if final.get(key) is not True:
            raise AssertionError(f"R6 method stage violated {key}: {method}")
    if final.get("paper_claim_authorized_by_method_stage") is not False:
        raise AssertionError(f"R6 method stage prematurely authorized a paper claim: {method}")
    if final.get("candidate_reeval_per_example_sha256") != common.file_sha256(rows_path):
        raise AssertionError(f"R6 method per-example hash mismatch: {method}")
    if final.get("candidate_reeval_summary_sha256") != common.file_sha256(summaries_path):
        raise AssertionError(f"R6 method summary CSV hash mismatch: {method}")
    if len(rows) != expected_rows or len(summaries) != 6:
        raise AssertionError(f"Unexpected R6 table dimensions: {method}")

    expected_draws = common.REEVAL_DRAWS if common.method_spec(method).stochastic else 1
    groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        if row.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError("R6 per-example protocol ID mismatch")
        if row.get("protocol_hash") != protocol["protocol_hash"]:
            raise AssertionError("R6 per-example protocol hash mismatch")
        if row.get("method") != method or row.get("attack") != common.METHOD_ATTACK[method]:
            raise AssertionError("R6 per-example method/attack mismatch")
        budget = int(row["api_budget"])
        seed = int(row["search_seed"])
        index = int(row["sample_index"])
        key = (budget, seed, index)
        if budget not in common.BUDGETS or seed not in common.SEARCH_SEEDS or key in seen:
            raise AssertionError("R6 budget/seed/index coverage or duplication error")
        seen.add(key)
        groups.setdefault((budget, seed), []).append(row)
        if int(row["eval_draws"]) != expected_draws:
            raise AssertionError("R6 MC draw count mismatch")
        if int(row["reeval_seed"]) != common.REEVAL_SEED:
            raise AssertionError("R6 reevaluation seed mismatch")
        for metric in PROBABILITY_METRICS:
            common.assert_probability(float(row[f"attacked_{metric}"]), f"attacked_{metric}")
            common.assert_probability(float(row[f"clean_{metric}"]), f"clean_{metric}")
        for metric in ("attacked_mean_boundary_margin", "clean_mean_boundary_margin"):
            if not np.isfinite(float(row[metric])):
                raise AssertionError(f"Non-finite R6 margin: {metric}")
    expected_groups = {
        (budget, seed) for budget in common.BUDGETS for seed in common.SEARCH_SEEDS
    }
    if set(groups) != expected_groups:
        raise AssertionError(f"R6 method group coverage mismatch: {method}")
    if any(len(group) != common.STRESS_EXAMPLES for group in groups.values()):
        raise AssertionError(f"R6 method group size mismatch: {method}")

    anchor_group = groups[(common.BUDGETS[0], common.SEARCH_SEEDS[0])]
    order = [int(row["sample_index"]) for row in anchor_group]
    if len(set(order)) != common.STRESS_EXAMPLES:
        raise AssertionError(f"R6 method sample duplication: {method}")
    clean_anchor = {int(row["sample_index"]): row for row in anchor_group}
    for group_key, group in groups.items():
        if [int(row["sample_index"]) for row in group] != order:
            raise AssertionError(f"R6 sample order mismatch: {method}/{group_key}")
        for row in group:
            anchor = clean_anchor[int(row["sample_index"])]
            for metric in (*PROBABILITY_METRICS, "mean_boundary_margin"):
                if float(row[f"clean_{metric}"]) != float(anchor[f"clean_{metric}"]):
                    raise AssertionError(f"R6 clean value changed across cells: {method}/{metric}")

    summary_keys = {(int(row["api_budget"]), int(row["search_seed"])) for row in summaries}
    if summary_keys != expected_groups:
        raise AssertionError(f"R6 candidate summary coverage mismatch: {method}")
    summary_map = {
        (int(row["api_budget"]), int(row["search_seed"])): row for row in summaries
    }
    for group_key, group in groups.items():
        summary = summary_map[group_key]
        checks = {
            "attacked_exact_set_probability_percent": 100.0
            * float(np.mean([float(row["attacked_exact_set_probability"]) for row in group])),
            "clean_exact_set_probability_percent": 100.0
            * float(np.mean([float(row["clean_exact_set_probability"]) for row in group])),
            "attacked_top1_accuracy_probability_percent": 100.0
            * float(np.mean([float(row["attacked_top1_accuracy_probability"]) for row in group])),
            "clean_top1_accuracy_probability_percent": 100.0
            * float(np.mean([float(row["clean_top1_accuracy_probability"]) for row in group])),
        }
        for key, expected in checks.items():
            if abs(float(summary[key]) - expected) > 1e-9:
                raise AssertionError(f"R6 aggregate mismatch: {method}/{group_key}/{key}")
    return rows, summaries


def validate_cross_method_alignment(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    signature_anchor: list[tuple[int, int, int, str]] | None = None
    indices: list[int] | None = None
    for method in common.METHODS:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and int(row["api_budget"]) == common.BUDGETS[0]
            and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
        ]
        subset.sort(key=lambda row: int(row["sample_index"]))
        signature = [
            (
                int(row["sample_index"]),
                int(row["label"]),
                int(row["reference_top1"]),
                str(row["reference_set"]),
            )
            for row in subset
        ]
        if signature_anchor is None:
            signature_anchor = signature
            indices = [item[0] for item in signature]
        elif signature != signature_anchor:
            raise AssertionError("R6 methods do not share samples, labels, and references")
    assert indices is not None
    return indices


def build_method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        for budget in common.BUDGETS:
            seed_exact: list[float] = []
            seed_top1: list[float] = []
            for seed in common.SEARCH_SEEDS:
                group = [
                    row
                    for row in rows
                    if row["method"] == method
                    and int(row["api_budget"]) == budget
                    and int(row["search_seed"]) == seed
                ]
                seed_exact.append(
                    100.0
                    * float(np.mean([float(row["attacked_exact_set_probability"]) for row in group]))
                )
                seed_top1.append(
                    100.0
                    * float(np.mean([float(row["attacked_top1_accuracy_probability"]) for row in group]))
                )
            indices, attacked_exact = common.aligned_values(
                rows, method=method, budget=budget, metric="attacked_exact_set_probability"
            )
            top1_indices, attacked_top1 = common.aligned_values(
                rows, method=method, budget=budget, metric="attacked_top1_accuracy_probability"
            )
            clean_indices, clean_exact = common.clean_values(
                rows, method=method, metric="clean_exact_set_probability"
            )
            clean_top1_indices, clean_top1 = common.clean_values(
                rows, method=method, metric="clean_top1_accuracy_probability"
            )
            if not (indices == top1_indices == clean_indices == clean_top1_indices):
                raise AssertionError("R6 method-summary metric alignment mismatch")
            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "method": method,
                    "attack": common.METHOD_ATTACK[method],
                    "api_budget": budget,
                    "examples": common.STRESS_EXAMPLES,
                    "search_seeds": "|".join(str(seed) for seed in common.SEARCH_SEEDS),
                    "attacked_exact_set_probability_percent": 100.0 * float(attacked_exact.mean()),
                    "attacked_exact_set_seed_sd_pp": float(statistics.stdev(seed_exact)),
                    "attacked_exact_set_seed_values_percent": "|".join(
                        f"{value:.6f}" for value in seed_exact
                    ),
                    "clean_exact_set_probability_percent": 100.0 * float(clean_exact.mean()),
                    "clean_to_attacked_exact_set_delta_pp": 100.0
                    * float((attacked_exact - clean_exact).mean()),
                    "attacked_top1_accuracy_probability_percent": 100.0
                    * float(attacked_top1.mean()),
                    "attacked_top1_seed_sd_pp": float(statistics.stdev(seed_top1)),
                    "attacked_top1_seed_values_percent": "|".join(
                        f"{value:.6f}" for value in seed_top1
                    ),
                    "clean_top1_accuracy_probability_percent": 100.0 * float(clean_top1.mean()),
                    "clean_to_attacked_top1_delta_pp": 100.0
                    * float((attacked_top1 - clean_top1).mean()),
                    "attack_family_frozen_from_r4": True,
                    "budget_run_independently": True,
                }
            )
    return output


def build_budget_envelope(
    rows: Sequence[Mapping[str, Any]], method_summary: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    selected: dict[str, int] = {}
    for method in common.METHODS:
        candidates = [row for row in method_summary if row["method"] == method]
        winner = min(
            candidates,
            key=lambda row: (
                float(row["attacked_exact_set_probability_percent"]),
                -int(row["api_budget"]),
            ),
        )
        selected_budget = int(winner["api_budget"])
        selected[method] = selected_budget
        q5 = next(row for row in candidates if int(row["api_budget"]) == 5000)
        q20 = next(row for row in candidates if int(row["api_budget"]) == 20000)
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "method": method,
                "attack": common.METHOD_ATTACK[method],
                "q5000_attacked_exact_set_percent": q5["attacked_exact_set_probability_percent"],
                "q20000_attacked_exact_set_percent": q20["attacked_exact_set_probability_percent"],
                "selected_budget": selected_budget,
                "cap_q_le_20000_attacked_exact_set_percent": winner[
                    "attacked_exact_set_probability_percent"
                ],
                "selection_unit": "whole_method",
                "per_example_selection_performed": False,
                "paper_claim_authorized": False,
            }
        )
    return output, selected


def build_comparisons(
    rows: Sequence[Mapping[str, Any]], selected_budget: Mapping[str, int]
) -> list[dict[str, Any]]:
    exact: dict[tuple[str, int], np.ndarray] = {}
    top1: dict[tuple[str, int], np.ndarray] = {}
    clean: dict[str, np.ndarray] = {}
    common_indices: list[int] | None = None
    for method in common.METHODS:
        clean_indices, clean[method] = common.clean_values(
            rows, method=method, metric="clean_exact_set_probability"
        )
        if common_indices is None:
            common_indices = clean_indices
        elif clean_indices != common_indices:
            raise AssertionError("R6 clean method alignment mismatch")
        for budget in common.BUDGETS:
            indices, exact[(method, budget)] = common.aligned_values(
                rows, method=method, budget=budget, metric="attacked_exact_set_probability"
            )
            top1_indices, top1[(method, budget)] = common.aligned_values(
                rows, method=method, budget=budget, metric="attacked_top1_accuracy_probability"
            )
            if indices != common_indices or top1_indices != common_indices:
                raise AssertionError("R6 comparison indices are not aligned")

    entries: list[tuple[str, str, str, int | str, np.ndarray]] = []
    for method in common.METHODS:
        entries.append(
            (
                f"{method}_q20000_minus_q5000_attacked_exact_set",
                "budget_delta",
                "paired_three_seed_mean",
                "20000_minus_5000",
                exact[(method, 20000)] - exact[(method, 5000)],
            )
        )
        entries.append(
            (
                f"{method}_q20000_minus_q5000_attacked_top1",
                "budget_delta_top1",
                "paired_three_seed_mean",
                "20000_minus_5000",
                top1[(method, 20000)] - top1[(method, 5000)],
            )
        )

    pairs = (
        ("full", "robust"),
        ("noise_only", "robust"),
        ("full_sigma0", "robust"),
        ("full", "noise_only"),
        ("full", "full_sigma0"),
    )
    for budget in common.BUDGETS:
        for left, right in pairs:
            entries.append(
                (
                    f"{left}_minus_{right}_q{budget}_attacked_exact_set",
                    "cross_method_raw_budget",
                    "paired_three_seed_mean",
                    budget,
                    exact[(left, budget)] - exact[(right, budget)],
                )
            )
        for left in ("full", "noise_only", "full_sigma0"):
            entries.append(
                (
                    f"{left}_minus_robust_q{budget}_attacked_top1",
                    "cross_method_raw_budget_top1",
                    "paired_three_seed_mean",
                    budget,
                    top1[(left, budget)] - top1[("robust", budget)],
                )
            )

    for method in common.METHODS:
        entries.append(
            (
                f"{method}_q20000_clean_to_attacked_exact_set",
                "clean_to_attack_q20000",
                "paired_three_seed_mean_minus_clean",
                20000,
                exact[(method, 20000)] - clean[method],
            )
        )
    full_drop = exact[("full", 20000)] - clean["full"]
    robust_drop = exact[("robust", 20000)] - clean["robust"]
    noise_drop = exact[("noise_only", 20000)] - clean["noise_only"]
    entries.extend(
        [
            (
                "full_vs_robust_q20000_difference_in_exact_set_drops",
                "difference_in_drops_q20000",
                "paired_three_seed_mean_minus_clean",
                20000,
                full_drop - robust_drop,
            ),
            (
                "noise_only_vs_robust_q20000_difference_in_exact_set_drops",
                "difference_in_drops_q20000",
                "paired_three_seed_mean_minus_clean",
                20000,
                noise_drop - robust_drop,
            ),
            (
                "full_vs_noise_only_q20000_difference_in_exact_set_drops",
                "difference_in_drops_q20000",
                "paired_three_seed_mean_minus_clean",
                20000,
                full_drop - noise_drop,
            ),
        ]
    )

    for left, right in pairs:
        left_budget = int(selected_budget[left])
        right_budget = int(selected_budget[right])
        entries.append(
            (
                f"{left}_minus_{right}_cap_q_le_20000_attacked_exact_set",
                "diagnostic_method_level_cap_envelope",
                "paired_selected_whole_method_budget_three_seed_mean",
                f"{left}:Q{left_budget}|{right}:Q{right_budget}",
                exact[(left, left_budget)] - exact[(right, right_budget)],
            )
        )

    output: list[dict[str, Any]] = []
    for name, family, estimand, budget, delta in entries:
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "comparison": name,
                "family": family,
                "estimand": estimand,
                "budget": budget,
                **common.paired_bootstrap(delta),
                "exploratory_post_confirmatory": True,
                "paper_claim_authorized": False,
            }
        )
    return output


def build_classwise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        clean_indices, clean = common.clean_values(
            rows, method=method, metric="clean_exact_set_probability"
        )
        anchor_rows = [
            row
            for row in rows
            if row["method"] == method
            and int(row["api_budget"]) == common.BUDGETS[0]
            and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
        ]
        labels = {int(row["sample_index"]): int(row["label"]) for row in anchor_rows}
        for budget in common.BUDGETS:
            indices, attacked = common.aligned_values(
                rows, method=method, budget=budget, metric="attacked_exact_set_probability"
            )
            if indices != clean_indices:
                raise AssertionError("R6 classwise clean/attack alignment mismatch")
            for class_id in range(100):
                positions = [pos for pos, index in enumerate(indices) if labels[index] == class_id]
                if len(positions) != common.STRESS_PER_CLASS:
                    raise AssertionError(f"R6 class {class_id} is not represented twice")
                attacked_value = float(attacked[positions].mean())
                clean_value = float(clean[positions].mean())
                output.append(
                    {
                        "protocol_id": common.PROTOCOL_ID,
                        "method": method,
                        "attack": common.METHOD_ATTACK[method],
                        "api_budget": budget,
                        "class_id": class_id,
                        "examples": common.STRESS_PER_CLASS,
                        "clean_exact_set_probability_percent": 100.0 * clean_value,
                        "attacked_exact_set_probability_percent": 100.0 * attacked_value,
                        "clean_to_attacked_exact_set_delta_pp": 100.0
                        * (attacked_value - clean_value),
                        "diagnostic_only": True,
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    common.load_r5_completion(Path(protocol["r5_root"]))
    output_dir = args.output_dir.expanduser().resolve()
    expected_dir = Path(protocol["output_root"]).expanduser().resolve() / "query_stress_analysis"
    if output_dir != expected_dir:
        raise ValueError("R6 analysis output directory differs from the frozen protocol")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    all_candidate_summaries: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, summaries = validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
        all_candidate_summaries.extend(summaries)
    indices = validate_cross_method_alignment(all_rows)
    if len(indices) != common.STRESS_EXAMPLES:
        raise AssertionError("R6 final analysis lacks 200 aligned examples")

    method_summary = build_method_summary(all_rows)
    envelope, selected_budget = build_budget_envelope(all_rows, method_summary)
    comparisons = build_comparisons(all_rows, selected_budget)
    classwise = build_classwise(all_rows)

    paths = {
        "query_stress_per_example.csv": output_dir / "query_stress_per_example.csv",
        "query_stress_candidate_summary.csv": output_dir / "query_stress_candidate_summary.csv",
        "query_stress_method_summary.csv": output_dir / "query_stress_method_summary.csv",
        "query_stress_paired_bootstrap.csv": output_dir / "query_stress_paired_bootstrap.csv",
        "query_stress_budget_envelope.csv": output_dir / "query_stress_budget_envelope.csv",
        "query_stress_classwise.csv": output_dir / "query_stress_classwise.csv",
    }
    common.atomic_csv(paths["query_stress_per_example.csv"], all_rows)
    common.atomic_csv(paths["query_stress_candidate_summary.csv"], all_candidate_summaries)
    common.atomic_csv(paths["query_stress_method_summary.csv"], method_summary)
    common.atomic_csv(paths["query_stress_paired_bootstrap.csv"], comparisons)
    common.atomic_csv(paths["query_stress_budget_envelope.csv"], envelope)
    common.atomic_csv(paths["query_stress_classwise.csv"], classwise)

    q20_full_robust = next(
        row
        for row in comparisons
        if row["comparison"] == "full_minus_robust_q20000_attacked_exact_set"
    )
    q20_noise_robust = next(
        row
        for row in comparisons
        if row["comparison"] == "noise_only_minus_robust_q20000_attacked_exact_set"
    )
    handoff: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "stage": "exploratory_post_confirmatory_query_budget_stress",
        "paper_primary_claim_authorized": False,
        "reason": "R6 was designed after observing R5 and is a robustness stress test",
        "q20000_full_minus_robust": q20_full_robust,
        "q20000_noise_only_minus_robust": q20_noise_robust,
        "next_stage": "ImageNet/architecture preflight after interpreting R6",
        "cross_attack_q20000_completion_performed": False,
        "no_per_example_budget_selection": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
    }
    handoff["handoff_hash"] = common.object_sha256(handoff)
    handoff_path = output_dir / "research_handoff.json"
    common.atomic_json(handoff_path, handoff)

    final = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "stage": "exploratory_post_confirmatory_query_budget_stress",
        "examples": common.STRESS_EXAMPLES,
        "methods": list(common.METHODS),
        "method_attack": dict(common.METHOD_ATTACK),
        "budgets": list(common.BUDGETS),
        "search_seeds": list(common.SEARCH_SEEDS),
        "epsilon_px": common.EPSILON_PX,
        "common_mc_draws_stochastic": common.REEVAL_DRAWS,
        "method_summary": method_summary,
        "budget_envelope": envelope,
        "q20000_full_minus_robust": q20_full_robust,
        "q20000_noise_only_minus_robust": q20_noise_robust,
        "paper_primary_claim_authorized": False,
        "r5_primary_claim_remains_the_confirmatory_claim": True,
        "defense_retuning_performed": False,
        "attack_family_selection_performed": False,
        "seed_selection_performed": False,
        "per_example_budget_selection_performed": False,
        "files": {
            name: common.file_sha256(path) for name, path in paths.items()
        }
        | {"research_handoff.json": common.file_sha256(handoff_path)},
    }
    final["final_summary_hash"] = common.object_sha256(final)
    common.atomic_json(output_dir / "final_summary.json", final)

    print("\nCIFAR-100 R6 query-budget stress results", flush=True)
    for method in common.METHODS:
        values = {
            int(row["api_budget"]): row
            for row in method_summary
            if row["method"] == method
        }
        delta = next(
            row
            for row in comparisons
            if row["comparison"] == f"{method}_q20000_minus_q5000_attacked_exact_set"
        )
        print(
            f"  {method:>11}: Q5k={float(values[5000]['attacked_exact_set_probability_percent']):.3f}% "
            f"Q20k={float(values[20000]['attacked_exact_set_probability_percent']):.3f}% "
            f"delta={float(delta['difference_pp']):+.3f} pp "
            f"[{float(delta['ci95_low_pp']):+.3f}, {float(delta['ci95_high_pp']):+.3f}]",
            flush=True,
        )
    print(
        f"Q20k Full-Robust={float(q20_full_robust['difference_pp']):+.3f} pp "
        f"[{float(q20_full_robust['ci95_low_pp']):+.3f}, "
        f"{float(q20_full_robust['ci95_high_pp']):+.3f}] -> "
        f"{q20_full_robust['status']}",
        flush=True,
    )
    print("PAPER_PRIMARY_CLAIM_AUTHORIZED=NO (exploratory post-confirmatory stress)", flush=True)
    print("NEXT_STAGE=INTERPRET_R6_THEN_IMAGENET_PREFLIGHT", flush=True)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

