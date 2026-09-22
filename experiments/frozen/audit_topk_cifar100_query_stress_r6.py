#!/usr/bin/env python3
"""Independent structural/statistical audit for CIFAR-100 R6 query stress."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import finalize_topk_cifar100_query_stress_r6 as finalize
import topk_cifar100_query_stress_r6_common as common
import topk_cifar100_square_attack_core_r4 as core
from search_topk_cifar100_query_stress_r6 import payload_identity, validate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CIFAR-100 query stress R6")
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--r5-root", type=Path, default=common.DEFAULT_R5_ROOT)
    parser.add_argument("--r3-parent-dir", type=Path, default=common.DEFAULT_R3_PARENT_DIR)
    return parser.parse_args()


def assert_close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(observed) or abs(observed - expected) > tolerance:
        raise AssertionError(f"{label}: observed={observed}, expected={expected}")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean: {value!r}")


def verify_protocol(
    output_root: Path, r5_root: Path, r3_parent_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
    protocol_path = output_root / "query_stress_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing frozen R6 protocol: {protocol_path}")
    protocol = common.read_json(protocol_path)
    common.validate_protocol(protocol)
    if Path(protocol["output_root"]).resolve() != output_root.resolve():
        raise AssertionError("R6 protocol output root differs from audit target")
    if Path(protocol["r5_root"]).resolve() != r5_root.resolve():
        raise AssertionError("R6 protocol R5 root differs from audit target")
    if Path(protocol["r3_parent_dir"]).resolve() != r3_parent_dir.resolve():
        raise AssertionError("R6 protocol R3 parent differs from audit target")
    for name in common.SCRIPT_NAMES:
        path = common.PROJECT_DIR / name
        if not path.is_file() or common.file_sha256(path) != protocol["script_sha256"].get(name):
            raise AssertionError(f"Frozen R6 source changed: {name}")
    for name in common.DEPENDENCY_NAMES:
        path = common.PROJECT_DIR / name
        if not path.is_file() or common.file_sha256(path) != protocol["dependency_sha256"].get(name):
            raise AssertionError(f"Frozen R6 dependency changed: {name}")

    r5_info = common.load_r5_completion(r5_root)
    if r5_info["protocol_hash"] != protocol["r5_protocol_hash"]:
        raise AssertionError("R5 protocol identity changed after R6 freeze")
    if r5_info["final_hash"] != protocol["r5_final_summary_hash"]:
        raise AssertionError("R5 result identity changed after R6 freeze")
    parent = common.r5.load_parent_authorization(r3_parent_dir)
    if parent["handoff_hash"] != protocol["r3_parent_handoff_hash"]:
        raise AssertionError("R3 parent changed after R6 freeze")
    derived = common.derive_stress_indices(parent)
    stress_indices = [int(value) for value in protocol["stress_indices"]]
    if stress_indices != derived:
        raise AssertionError("R6 stress split differs from deterministic label-only selection")
    common.validate_stress_indices(stress_indices, parent)

    manifest_path = output_root / "stress_split_manifest.json"
    snapshot_path = output_root / "r5_completion_snapshot.json"
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("R6 split/R5 identity snapshots are incomplete")
    manifest = common.read_json(manifest_path)
    claimed = str(manifest.get("manifest_hash", ""))
    base = dict(manifest)
    base.pop("manifest_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError("R6 stress split manifest self-hash mismatch")
    if manifest.get("indices") != stress_indices:
        raise AssertionError("R6 split manifest indices differ from protocol")
    if manifest.get("attack_outcomes_read_for_selection") is not False:
        raise AssertionError("R6 split manifest reports outcome-based selection")
    snapshot = common.read_json(snapshot_path)
    claimed = str(snapshot.get("snapshot_hash", ""))
    base = dict(snapshot)
    base.pop("snapshot_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError("R6 R5 snapshot self-hash mismatch")
    if snapshot.get("r5_final_summary_hash") != r5_info["final_hash"]:
        raise AssertionError("R6 R5 snapshot points to a different result")
    return protocol, parent, r5_info, stress_indices


def expected_query_details(method: str, budget: int) -> Mapping[str, Any]:
    attack = common.METHOD_ATTACK[method]
    stochastic = common.method_spec(method).stochastic
    if attack == "SquareAdaptive":
        plan = core.square_adaptive_query_plan(
            budget,
            int(common.ADAPTIVE_CONFIG["search_eot"]),
            int(common.ADAPTIVE_CONFIG["refresh_eot"]),
            int(common.ADAPTIVE_CONFIG["confirm_candidates"]),
            float(common.ADAPTIVE_CONFIG["confirm_fraction"]),
            stochastic,
        )
        return {
            "search_steps": plan["search_steps"],
            "search_eot": common.ADAPTIVE_CONFIG["search_eot"],
            "refresh_eot": common.ADAPTIVE_CONFIG["refresh_eot"],
            "confirmation_candidates": common.ADAPTIVE_CONFIG["confirm_candidates"],
            "confirmation_queries": plan["confirmation_queries"],
            "confirmation_draws_min": min(plan["confirmation_draws"]),
            "confirmation_draws_max": max(plan["confirmation_draws"]),
            "logical_calls": plan["logical_calls"],
            "api_queries": budget,
            "score_objective": common.ADAPTIVE_CONFIG["objective"],
        }
    if stochastic:
        search_eot = int(common.RACING_CONFIG["search_eot"])
        refresh_eot = int(common.RACING_CONFIG["refresh_eot"])
        pool_size = int(common.RACING_CONFIG["pool_size"])
        race_interval = int(common.RACING_CONFIG["race_interval"])
        race_eot = int(common.RACING_CONFIG["race_eot"])
    else:
        search_eot, refresh_eot, pool_size = 1, 0, 1
        race_interval, race_eot = budget + 1, 0
    plan = core.square_racing_query_plan(
        budget,
        search_eot,
        refresh_eot,
        pool_size,
        race_interval,
        race_eot,
        float(common.RACING_CONFIG["confirm_fraction"]),
        stochastic,
    )
    return {
        "search_steps": plan["search_steps"],
        "search_eot": search_eot,
        "refresh_eot": refresh_eot,
        "race_rounds": plan["race_rounds"],
        "race_interval": race_interval,
        "race_eot": race_eot,
        "race_queries": plan["race_queries"],
        "confirmation_candidates": pool_size,
        "confirmation_queries": plan["confirmation_queries"],
        "confirmation_draws_min": min(plan["confirmation_draws"]),
        "confirmation_draws_max": max(plan["confirmation_draws"]),
        "logical_calls": plan["logical_calls"],
        "api_queries": budget,
        "score_temperature": common.RACING_CONFIG["score_temperature"],
        "score_objective": (
            common.RACING_CONFIG["objective"]
            if stochastic
            else "deterministic_exact_set_margin"
        ),
    }


def verify_search_cells(
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    output_root: Path,
    stress_indices: Sequence[int],
) -> list[dict[str, Any]]:
    expected_labels, expected_references = common.load_stress_references(parent, stress_indices)
    records: list[dict[str, Any]] = []
    shared: dict[str, str] | None = None
    for method, attack, budget, seed in common.expected_cells():
        directory = common.cell_dir(output_root, method, budget, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        rows_path = directory / "search_per_example.csv"
        missing = [str(path) for path in (candidate_path, summary_path, rows_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete R6 search cell:\n  " + "\n  ".join(missing))
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        identity = payload_identity(
            protocol_hash=str(protocol["protocol_hash"]),
            method=method,
            budget=budget,
            search_seed=seed,
        )
        validate_payload(payload, identity, stress_indices)
        if payload.get("query_details") != expected_query_details(method, budget):
            raise AssertionError(f"Unexpected R6 query plan: {method}/Q{budget}/{seed}")
        if payload.get("r5_final_summary_hash") != protocol["r5_final_summary_hash"]:
            raise AssertionError("R6 candidate R5 identity mismatch")
        summary = common.read_json(summary_path)
        if summary.get("status") != "PASS" or int(summary.get("examples", -1)) != 200:
            raise AssertionError(f"Invalid R6 search summary: {directory}")
        if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
            raise AssertionError(f"R6 candidate file hash mismatch: {directory}")
        if summary.get("r4_r5_partitions_read") is not False:
            raise AssertionError("R6 search reports attack access to R4/R5 samples")
        rows = common.read_csv(rows_path)
        if len(rows) != common.STRESS_EXAMPLES:
            raise AssertionError(f"R6 search row count mismatch: {directory}")
        if [int(row["sample_index"]) for row in rows] != list(stress_indices):
            raise AssertionError(f"R6 search row order mismatch: {directory}")
        if any(int(row["api_budget"]) != budget for row in rows):
            raise AssertionError("R6 per-example query count differs from cell budget")
        hashes = payload["tensor_sha256"]
        anchors = {
            key: hashes[key] for key in ("clean", "labels", "sample_indices", "reference_top2")
        }
        if shared is None:
            shared = anchors
        elif anchors != shared:
            raise AssertionError("R6 search cells do not share clean anchors")
        records.append(
            {
                "method": method,
                "attack": attack,
                "budget": budget,
                "seed": seed,
                "payload": payload,
            }
        )
    first = records[0]["payload"]
    if [int(value) for value in first["labels"].tolist()] != expected_labels:
        raise AssertionError("R6 labels differ from audited R3 rows")
    references = [[int(value) for value in row] for row in first["reference_top2"].tolist()]
    if references != expected_references:
        raise AssertionError("R6 references differ from audited R3 rows")
    return records


def _row_map(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    output: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        if key in output:
            raise AssertionError(f"Duplicate aggregate key: {key}")
        output[key] = row
    return output


def verify_analysis(
    protocol: Mapping[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, str]] = []
    all_candidate_summaries: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, summaries = finalize.validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
        all_candidate_summaries.extend(summaries)
    if len(all_rows) != 4800 or len(all_candidate_summaries) != 24:
        raise AssertionError("R6 reevaluation table dimensions changed")
    finalize.validate_cross_method_alignment(all_rows)
    expected_method = finalize.build_method_summary(all_rows)
    expected_envelope, selected = finalize.build_budget_envelope(all_rows, expected_method)
    expected_comparisons = finalize.build_comparisons(all_rows, selected)
    expected_classwise = finalize.build_classwise(all_rows)

    directory = output_root / "query_stress_analysis"
    paths = {
        "per_example": directory / "query_stress_per_example.csv",
        "candidate_summary": directory / "query_stress_candidate_summary.csv",
        "method_summary": directory / "query_stress_method_summary.csv",
        "comparisons": directory / "query_stress_paired_bootstrap.csv",
        "envelope": directory / "query_stress_budget_envelope.csv",
        "classwise": directory / "query_stress_classwise.csv",
        "handoff": directory / "research_handoff.json",
        "final": directory / "final_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete R6 analysis:\n  " + "\n  ".join(missing))
    if common.read_csv(paths["per_example"]) != all_rows:
        raise AssertionError("R6 consolidated per-example rows differ from method tables")
    if common.read_csv(paths["candidate_summary"]) != all_candidate_summaries:
        raise AssertionError("R6 consolidated candidate summaries differ from method tables")
    saved_method = common.read_csv(paths["method_summary"])
    saved_comparisons = common.read_csv(paths["comparisons"])
    saved_envelope = common.read_csv(paths["envelope"])
    saved_classwise = common.read_csv(paths["classwise"])
    if (len(saved_method), len(saved_comparisons), len(saved_envelope), len(saved_classwise)) != (
        8,
        36,
        4,
        800,
    ):
        raise AssertionError("R6 final table dimensions changed")

    observed = _row_map(saved_method, ("method", "api_budget"))
    for expected in expected_method:
        row = observed.get((str(expected["method"]), str(expected["api_budget"])))
        if row is None or row["attack"] != expected["attack"]:
            raise AssertionError("R6 method summary identity mismatch")
        for key in (
            "attacked_exact_set_probability_percent",
            "attacked_exact_set_seed_sd_pp",
            "clean_exact_set_probability_percent",
            "clean_to_attacked_exact_set_delta_pp",
            "attacked_top1_accuracy_probability_percent",
            "clean_top1_accuracy_probability_percent",
        ):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"R6 method {key}")

    observed = _row_map(saved_comparisons, ("comparison",))
    for expected in expected_comparisons:
        row = observed.get((str(expected["comparison"]),))
        if row is None or row["family"] != expected["family"]:
            raise AssertionError("R6 comparison identity mismatch")
        for key in ("difference_pp", "ci95_low_pp", "ci95_high_pp"):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"R6 comparison {key}")
        if as_bool(row["paper_claim_authorized"]):
            raise AssertionError("R6 comparison incorrectly authorizes a paper primary claim")

    observed = _row_map(saved_envelope, ("method",))
    for expected in expected_envelope:
        row = observed.get((str(expected["method"]),))
        if row is None or int(row["selected_budget"]) != int(expected["selected_budget"]):
            raise AssertionError("R6 method-level budget envelope mismatch")
        if as_bool(row["per_example_selection_performed"]):
            raise AssertionError("R6 envelope reports forbidden per-example selection")

    observed = _row_map(saved_classwise, ("method", "api_budget", "class_id"))
    for expected in expected_classwise:
        key = (str(expected["method"]), str(expected["api_budget"]), str(expected["class_id"]))
        row = observed.get(key)
        if row is None:
            raise AssertionError("R6 classwise cell missing")
        for metric in (
            "clean_exact_set_probability_percent",
            "attacked_exact_set_probability_percent",
            "clean_to_attacked_exact_set_delta_pp",
        ):
            assert_close(float(row[metric]), float(expected[metric]), 1e-9, f"R6 classwise {metric}")

    handoff = common.read_json(paths["handoff"])
    claimed = str(handoff.get("handoff_hash", ""))
    base = dict(handoff)
    base.pop("handoff_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError("R6 research handoff self-hash mismatch")
    if handoff.get("paper_primary_claim_authorized") is not False:
        raise AssertionError("R6 handoff incorrectly authorizes a primary claim")
    if handoff.get("no_per_example_budget_selection") is not True:
        raise AssertionError("R6 handoff permits per-example budget selection")

    final = common.read_json(paths["final"])
    claimed = str(final.get("final_summary_hash", ""))
    base = dict(final)
    base.pop("final_summary_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError("R6 final-summary self-hash mismatch")
    if final.get("protocol_hash") != protocol["protocol_hash"] or final.get("status") != "PASS":
        raise AssertionError("R6 final/protocol identity or status mismatch")
    if final.get("paper_primary_claim_authorized") is not False:
        raise AssertionError("R6 final incorrectly authorizes a primary paper claim")
    for key in (
        "defense_retuning_performed",
        "attack_family_selection_performed",
        "seed_selection_performed",
        "per_example_budget_selection_performed",
    ):
        if final.get(key) is not False:
            raise AssertionError(f"R6 final reports forbidden action: {key}")
    for name, expected_hash in final.get("files", {}).items():
        path = directory / name
        if not path.is_file() or common.file_sha256(path) != expected_hash:
            raise AssertionError(f"R6 final file hash mismatch: {name}")
    return expected_method, expected_comparisons


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    r5_root = args.r5_root.expanduser().resolve()
    r3_parent_dir = args.r3_parent_dir.expanduser().resolve()
    protocol, parent, _, stress_indices = verify_protocol(output_root, r5_root, r3_parent_dir)
    records = verify_search_cells(protocol, parent, output_root, stress_indices)
    method_summary, comparisons = verify_analysis(protocol, output_root)
    print(
        f"PASS {output_root}: {len(records)} exact-budget cells, "
        "4,800 common-MC rows, candidates feasible",
        flush=True,
    )
    print("CIFAR-100 R6 Q=5k/20k stress summary", flush=True)
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
    primary_diagnostic = next(
        row
        for row in comparisons
        if row["comparison"] == "full_minus_robust_q20000_attacked_exact_set"
    )
    print(
        f"  Q20k Full-Robust={float(primary_diagnostic['difference_pp']):+.3f} pp "
        f"[{float(primary_diagnostic['ci95_low_pp']):+.3f}, "
        f"{float(primary_diagnostic['ci95_high_pp']):+.3f}] -> "
        f"{primary_diagnostic['status']}",
        flush=True,
    )
    print("PASS split is balanced and disjoint from R4/R5; no retuning or selection", flush=True)
    print("PAPER_PRIMARY_CLAIM_AUTHORIZED=NO (exploratory stress only)", flush=True)
    print("R6 audit complete; no merge, deletion, training, retuning, or selection was performed.", flush=True)


if __name__ == "__main__":
    main()

