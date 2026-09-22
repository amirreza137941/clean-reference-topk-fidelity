#!/usr/bin/env python3
"""Preregistered analysis of all four completed CIFAR-100 R5 methods."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_cifar100_confirmatory_r5_common as common


PROBABILITY_METRICS = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "exact_set_probability",
    "label_coverage_probability",
    "mean_logits_top1_accuracy",
    "mean_logits_exact_set",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize CIFAR-100 confirmatory R5")
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
    directory = common.method_reeval_dir(Path(protocol["output_root"]), method)
    rows_path = directory / "candidate_reeval_per_example.csv"
    summaries_path = directory / "candidate_reeval_summary.csv"
    final_path = directory / "method_summary.json"
    missing = [str(path) for path in (rows_path, summaries_path, final_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete R5 method reevaluation:\n  " + "\n  ".join(missing))
    rows = common.read_csv(rows_path)
    summaries = common.read_csv(summaries_path)
    final = common.read_json(final_path)
    claimed = str(final.get("method_summary_hash", ""))
    base = dict(final)
    base.pop("method_summary_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError(f"R5 method-summary hash mismatch: {method}")
    if final.get("protocol_hash") != protocol["protocol_hash"]:
        raise AssertionError(f"R5 method/protocol identity mismatch: {method}")
    if final.get("method") != method or final.get("attack") != common.METHOD_ATTACK[method]:
        raise AssertionError(f"R5 method/attack identity mismatch: {method}")
    if final.get("status") != "PASS":
        raise AssertionError(f"R5 method stage did not pass: {method}")
    if final.get("search_seeds") != list(common.SEARCH_SEEDS):
        raise AssertionError(f"R5 method seeds changed: {method}")
    if int(final.get("examples", -1)) != 1000 or int(final.get("per_example_rows", -1)) != 3000:
        raise AssertionError(f"R5 method dimensions changed: {method}")
    if final.get("no_attack_family_selection") is not True or final.get("no_seed_selection") is not True:
        raise AssertionError(f"R5 method stage performed forbidden selection: {method}")
    if final.get("paper_claim_authorized_by_method_stage") is not False:
        raise AssertionError(f"R5 method stage prematurely authorized a claim: {method}")
    if final.get("candidate_reeval_per_example_sha256") != common.file_sha256(rows_path):
        raise AssertionError(f"R5 method per-example file hash mismatch: {method}")
    if final.get("candidate_reeval_summary_sha256") != common.file_sha256(summaries_path):
        raise AssertionError(f"R5 method summary CSV hash mismatch: {method}")
    if len(rows) != 3000 or len(summaries) != 3:
        raise AssertionError(f"Unexpected R5 method table dimensions: {method}")

    expected_draws = common.REEVAL_DRAWS if common.method_spec(method).stochastic else 1
    seen: set[tuple[int, int]] = set()
    groups: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError("R5 per-example protocol ID mismatch")
        if row.get("protocol_hash") != protocol["protocol_hash"]:
            raise AssertionError("R5 per-example protocol hash mismatch")
        if row.get("method") != method or row.get("attack") != common.METHOD_ATTACK[method]:
            raise AssertionError("R5 per-example method/attack mismatch")
        seed = int(row["search_seed"])
        index = int(row["sample_index"])
        if seed not in common.SEARCH_SEEDS or (seed, index) in seen:
            raise AssertionError("R5 per-example seed/index duplication")
        seen.add((seed, index))
        groups.setdefault(seed, []).append(row)
        if int(row["eval_draws"]) != expected_draws:
            raise AssertionError("R5 per-example MC draw count mismatch")
        if int(row["reeval_seed"]) != common.REEVAL_SEED:
            raise AssertionError("R5 per-example reevaluation seed mismatch")
        if int(row["api_budget"]) != common.API_BUDGET:
            raise AssertionError("R5 per-example API budget mismatch")
        for metric in PROBABILITY_METRICS:
            common.assert_probability(float(row[f"attacked_{metric}"]), f"attacked_{metric}")
            common.assert_probability(float(row[f"clean_{metric}"]), f"clean_{metric}")
        for metric in ("attacked_mean_boundary_margin", "clean_mean_boundary_margin"):
            if not np.isfinite(float(row[metric])):
                raise AssertionError(f"Non-finite R5 boundary margin: {metric}")
    if set(groups) != set(common.SEARCH_SEEDS) or any(len(group) != 1000 for group in groups.values()):
        raise AssertionError(f"R5 method seed coverage mismatch: {method}")
    order = [int(row["sample_index"]) for row in groups[common.SEARCH_SEEDS[0]]]
    if len(set(order)) != 1000:
        raise AssertionError(f"R5 method sample duplication: {method}")
    for seed, group in groups.items():
        if [int(row["sample_index"]) for row in group] != order:
            raise AssertionError(f"R5 seed sample order mismatch: {method}/{seed}")
    first_by_index = {
        int(row["sample_index"]): row for row in groups[common.SEARCH_SEEDS[0]]
    }
    for seed in common.SEARCH_SEEDS[1:]:
        for row in groups[seed]:
            anchor = first_by_index[int(row["sample_index"])]
            for metric in (*PROBABILITY_METRICS, "mean_boundary_margin"):
                if float(row[f"clean_{metric}"]) != float(anchor[f"clean_{metric}"]):
                    raise AssertionError(
                        f"R5 clean reevaluation changed across seeds: {method}/{metric}"
                    )
    summary_keys = {(int(row["search_seed"]), row["attack"]) for row in summaries}
    expected_keys = {(seed, common.METHOD_ATTACK[method]) for seed in common.SEARCH_SEEDS}
    if summary_keys != expected_keys:
        raise AssertionError(f"R5 candidate summary coverage mismatch: {method}")
    summary_map = {int(row["search_seed"]): row for row in summaries}
    for seed, group in groups.items():
        summary = summary_map[seed]
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
            observed = float(summary[key])
            if not np.isfinite(observed) or abs(observed - expected) > 1e-9:
                raise AssertionError(
                    f"R5 candidate summary does not match rows: {method}/{seed}/{key}"
                )
    return rows, summaries


def validate_cross_method_alignment(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    reference: list[tuple[int, int, int, str]] | None = None
    indices: list[int] | None = None
    for method in common.METHODS:
        subset = [
            row for row in rows
            if row["method"] == method and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
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
        if reference is None:
            reference = signature
            indices = [value[0] for value in signature]
        elif signature != reference:
            raise AssertionError("R5 methods do not share samples, labels, and references")
    assert indices is not None
    return indices


def build_method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        attack = common.METHOD_ATTACK[method]
        seed_exact: list[float] = []
        seed_top1: list[float] = []
        for seed in common.SEARCH_SEEDS:
            group = [
                row for row in rows
                if row["method"] == method and int(row["search_seed"]) == seed
            ]
            if len(group) != 1000:
                raise AssertionError(f"Missing R5 group for {method}/{seed}")
            seed_exact.append(
                100.0 * float(np.mean([float(row["attacked_exact_set_probability"]) for row in group]))
            )
            seed_top1.append(
                100.0 * float(np.mean([float(row["attacked_top1_accuracy_probability"]) for row in group]))
            )
        _, attacked_exact = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_probability"
        )
        _, attacked_top1 = common.aligned_values(
            rows, method=method, metric="attacked_top1_accuracy_probability"
        )
        _, clean_exact = common.clean_values(
            rows, method=method, metric="clean_exact_set_probability"
        )
        _, clean_top1 = common.clean_values(
            rows, method=method, metric="clean_top1_accuracy_probability"
        )
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "method": method,
                "attack": attack,
                "examples": 1000,
                "search_seeds": "|".join(str(seed) for seed in common.SEARCH_SEEDS),
                "attacked_exact_set_probability_percent": 100.0 * float(attacked_exact.mean()),
                "attacked_exact_set_seed_sd_pp": float(statistics.stdev(seed_exact)),
                "attacked_exact_set_seed_values_percent": "|".join(
                    f"{value:.6f}" for value in seed_exact
                ),
                "clean_exact_set_probability_percent": 100.0 * float(clean_exact.mean()),
                "clean_to_attacked_exact_set_delta_pp": 100.0
                * float((attacked_exact - clean_exact).mean()),
                "attacked_top1_accuracy_probability_percent": 100.0 * float(attacked_top1.mean()),
                "attacked_top1_seed_sd_pp": float(statistics.stdev(seed_top1)),
                "attacked_top1_seed_values_percent": "|".join(
                    f"{value:.6f}" for value in seed_top1
                ),
                "clean_top1_accuracy_probability_percent": 100.0 * float(clean_top1.mean()),
                "clean_to_attacked_top1_delta_pp": 100.0
                * float((attacked_top1 - clean_top1).mean()),
                "attack_family_frozen_on_disjoint_pilot": True,
                "seed_selection_performed": False,
            }
        )
    return output


def build_classwise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        indices, attacked_exact = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_probability"
        )
        clean_indices, clean_exact = common.clean_values(
            rows, method=method, metric="clean_exact_set_probability"
        )
        if clean_indices != indices:
            raise AssertionError("R5 classwise clean/attack index alignment mismatch")
        first_seed = [
            row for row in rows
            if row["method"] == method and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
        ]
        labels = {int(row["sample_index"]): int(row["label"]) for row in first_seed}
        for class_id in range(100):
            positions = [pos for pos, index in enumerate(indices) if labels[index] == class_id]
            if len(positions) != 10:
                raise AssertionError(f"R5 class {class_id} is not represented by 10 examples")
            attacked = float(attacked_exact[positions].mean())
            clean = float(clean_exact[positions].mean())
            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "method": method,
                    "attack": common.METHOD_ATTACK[method],
                    "class_id": class_id,
                    "examples": 10,
                    "clean_exact_set_probability_percent": 100.0 * clean,
                    "attacked_exact_set_probability_percent": 100.0 * attacked,
                    "clean_to_attacked_exact_set_delta_pp": 100.0 * (attacked - clean),
                }
            )
    return output


def build_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    attacked_exact: dict[str, np.ndarray] = {}
    attacked_top1: dict[str, np.ndarray] = {}
    clean_exact: dict[str, np.ndarray] = {}
    common_indices: list[int] | None = None
    for method in common.METHODS:
        indices, attacked_exact[method] = common.aligned_values(
            rows, method=method, metric="attacked_exact_set_probability"
        )
        top1_indices, attacked_top1[method] = common.aligned_values(
            rows, method=method, metric="attacked_top1_accuracy_probability"
        )
        clean_indices, clean_exact[method] = common.clean_values(
            rows, method=method, metric="clean_exact_set_probability"
        )
        if indices != top1_indices or indices != clean_indices:
            raise AssertionError("R5 comparison metric indices are not aligned")
        if common_indices is None:
            common_indices = indices
        elif indices != common_indices:
            raise AssertionError("R5 comparison methods are not aligned")

    primary_secondary = (
        (common.PRIMARY_COMPARISON, "primary", attacked_exact["full"] - attacked_exact["robust"]),
        (
            "noise_only_minus_robust_attacked_exact_set",
            "secondary",
            attacked_exact["noise_only"] - attacked_exact["robust"],
        ),
        (
            "full_sigma0_minus_robust_attacked_exact_set",
            "secondary",
            attacked_exact["full_sigma0"] - attacked_exact["robust"],
        ),
        (
            "full_minus_noise_only_attacked_exact_set",
            "secondary",
            attacked_exact["full"] - attacked_exact["noise_only"],
        ),
        (
            "full_minus_full_sigma0_attacked_exact_set",
            "secondary",
            attacked_exact["full"] - attacked_exact["full_sigma0"],
        ),
    )
    diagnostic: list[tuple[str, str, np.ndarray]] = []
    drops = {method: attacked_exact[method] - clean_exact[method] for method in common.METHODS}
    for method in common.METHODS:
        diagnostic.append(
            (
                f"{method}_clean_to_attacked_exact_set",
                "diagnostic",
                drops[method],
            )
        )
    diagnostic.extend(
        [
            (
                "full_vs_robust_difference_in_exact_set_drops",
                "secondary",
                drops["full"] - drops["robust"],
            ),
            (
                "noise_only_vs_robust_difference_in_exact_set_drops",
                "secondary",
                drops["noise_only"] - drops["robust"],
            ),
            (
                "full_sigma0_vs_robust_difference_in_exact_set_drops",
                "secondary",
                drops["full_sigma0"] - drops["robust"],
            ),
            (
                "full_vs_noise_only_difference_in_exact_set_drops",
                "secondary",
                drops["full"] - drops["noise_only"],
            ),
            (
                "full_minus_robust_attacked_top1",
                "secondary_top1",
                attacked_top1["full"] - attacked_top1["robust"],
            ),
            (
                "noise_only_minus_robust_attacked_top1",
                "secondary_top1",
                attacked_top1["noise_only"] - attacked_top1["robust"],
            ),
            (
                "full_sigma0_minus_robust_attacked_top1",
                "secondary_top1",
                attacked_top1["full_sigma0"] - attacked_top1["robust"],
            ),
        ]
    )
    output: list[dict[str, Any]] = []
    for name, family, delta in (*primary_secondary, *diagnostic):
        result = common.paired_bootstrap(delta)
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "comparison": name,
                "family": family,
                "estimand": "confirmatory_fixed_attack_three_seed_mean_per_example",
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
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir != Path(protocol["output_root"]).expanduser().resolve() / "confirmatory_analysis":
        raise ValueError("R5 final analysis output directory differs from the frozen protocol")

    all_rows: list[dict[str, str]] = []
    all_candidate_summaries: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, summaries = validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
        all_candidate_summaries.extend(summaries)
    indices = validate_cross_method_alignment(all_rows)
    if len(indices) != 1000:
        raise AssertionError("R5 final analysis does not contain 1,000 aligned examples")

    method_summary = build_method_summary(all_rows)
    comparisons = build_comparisons(all_rows)
    classwise = build_classwise(all_rows)
    primary = next(row for row in comparisons if row["comparison"] == common.PRIMARY_COMPARISON)
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
            "On the preregistered disjoint CIFAR-100 ResNet-20 holdout, the frozen "
            "full defense achieved higher attacked Exact Top-2 set probability than "
            "the frozen robust anchor under method-specific attacks selected only on "
            "the disjoint pilot."
            if primary_pass
            else None
        ),
    }
    handoff["handoff_hash"] = common.object_sha256(handoff)
    common.atomic_json(output_dir / "paper_claim_handoff.json", handoff)

    final = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "examples": 1000,
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
        "defense_retuning_performed": False,
        "files": {
            "confirmatory_per_example.csv": common.file_sha256(per_example_path),
            "confirmatory_candidate_summary.csv": common.file_sha256(candidate_summary_path),
            "confirmatory_method_summary.csv": common.file_sha256(method_summary_path),
            "confirmatory_paired_bootstrap.csv": common.file_sha256(comparison_path),
            "confirmatory_classwise.csv": common.file_sha256(classwise_path),
            "paper_claim_handoff.json": common.file_sha256(output_dir / "paper_claim_handoff.json"),
        },
    }
    final["final_summary_hash"] = common.object_sha256(final)
    common.atomic_json(output_dir / "final_summary.json", final)

    print("\nCIFAR-100 R5 preregistered confirmatory results", flush=True)
    for row in method_summary:
        print(
            f"  {row['method']:>11}: attack={row['attack']} "
            f"clean Exact={float(row['clean_exact_set_probability_percent']):.3f}% "
            f"attacked Exact={float(row['attacked_exact_set_probability_percent']):.3f}% "
            f"sd={float(row['attacked_exact_set_seed_sd_pp']):.3f} pp",
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
