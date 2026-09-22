#!/usr/bin/env python3
"""Structural/statistical audit for the complete CIFAR-100 attack pilot R4."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import topk_cifar100_attack_pilot_r4_common as common
import topk_cifar100_square_attack_core_r4 as core
from reevaluate_topk_cifar100_attack_pilot_r4_mc1000 import (
    family_and_selection,
    pilot_comparisons,
)
from search_topk_cifar100_attack_pilot_r4 import payload_identity, validate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CIFAR-100 adaptive attack pilot R4")
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parent-dir", type=Path, default=common.DEFAULT_PARENT_DIR)
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


def verify_protocol(output_root: Path, parent_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = output_root / "pilot_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing frozen protocol: {protocol_path}")
    protocol = common.read_json(protocol_path)
    protocol_hash = common.validate_protocol(protocol)
    if Path(protocol["output_root"]).resolve() != output_root.resolve():
        raise AssertionError("Protocol output root differs from audit target")
    if Path(protocol["parent_dir"]).resolve() != parent_dir.resolve():
        raise AssertionError("Protocol parent directory differs from audit target")
    for name in common.SCRIPT_NAMES:
        path = common.PROJECT_DIR / name
        if not path.is_file() or common.file_sha256(path) != protocol["script_sha256"].get(name):
            raise AssertionError(f"Frozen R4 source changed: {name}")
    parent = common.load_parent_authorization(parent_dir)
    if protocol["parent_handoff_hash"] != parent["handoff_hash"]:
        raise AssertionError("R3 handoff changed after R4 freeze")
    if protocol["pilot_indices_sha256"] != common.sequence_sha256(parent["pilot_indices"]):
        raise AssertionError("R4 pilot index hash differs from R3")
    if protocol["confirmatory_indices_sha256"] != common.sequence_sha256(
        parent["confirmatory_indices"]
    ):
        raise AssertionError("R4 confirmatory index hash differs from R3")
    return protocol, parent


def expected_query_details(method: str, attack: str) -> Mapping[str, Any]:
    stochastic = common.method_spec(method).stochastic
    if attack == "SquareAdaptive":
        plan = core.square_adaptive_query_plan(
            common.API_BUDGET,
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
            "api_queries": common.API_BUDGET,
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
        race_interval, race_eot = common.API_BUDGET + 1, 0
    plan = core.square_racing_query_plan(
        common.API_BUDGET,
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
        "api_queries": common.API_BUDGET,
        "score_temperature": common.RACING_CONFIG["score_temperature"],
        "score_objective": (
            common.RACING_CONFIG["objective"]
            if stochastic
            else "deterministic_exact_set_margin"
        ),
    }


def verify_search_cells(
    protocol: Mapping[str, Any], parent: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    pilot = [int(value) for value in parent["pilot_indices"]]
    confirmatory = set(int(value) for value in parent["confirmatory_indices"])
    records: list[dict[str, Any]] = []
    anchor_hashes: dict[str, str] | None = None
    for method, attack, seed in common.expected_cells():
        directory = common.cell_dir(output_root, method, attack, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        rows_path = directory / "search_per_example.csv"
        missing = [str(path) for path in (candidate_path, summary_path, rows_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete search cell:\n  " + "\n  ".join(missing))
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        identity = payload_identity(
            protocol_hash=str(protocol["protocol_hash"]),
            method=method,
            attack=attack,
            search_seed=seed,
        )
        validate_payload(payload, identity, pilot)
        if set(int(value) for value in payload["sample_indices"].tolist()) & confirmatory:
            raise AssertionError("A search candidate includes confirmatory examples")
        if payload["query_details"] != expected_query_details(method, attack):
            raise AssertionError(f"Unexpected query plan: {method}/{attack}/{seed}")
        summary = common.read_json(summary_path)
        if summary.get("status") != "PASS" or int(summary.get("examples", -1)) != 200:
            raise AssertionError(f"Invalid search summary: {directory}")
        if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
            raise AssertionError(f"Candidate file hash mismatch: {directory}")
        if as_bool(summary.get("confirmatory_partition_read")):
            raise AssertionError("Search summary claims confirmatory partition was read")
        if int(summary["query_details"]["api_queries"]) != common.API_BUDGET:
            raise AssertionError("Search did not consume exact Q=5000")
        rows = common.read_csv(rows_path)
        if len(rows) != 200 or [int(row["sample_index"]) for row in rows] != pilot:
            raise AssertionError(f"Search per-example rows misalign: {directory}")
        if any(int(row["api_queries"]) != common.API_BUDGET for row in rows):
            raise AssertionError("Per-example query counts differ from Q=5000")
        hashes = payload["tensor_sha256"]
        shared = {key: hashes[key] for key in ("clean", "labels", "sample_indices", "reference_top2")}
        if anchor_hashes is None:
            anchor_hashes = shared
        elif shared != anchor_hashes:
            raise AssertionError("Candidate cells do not share clean data/labels/references")
        records.append(
            {
                "method": method,
                "attack": attack,
                "search_seed": seed,
                "candidate_path": candidate_path,
                "payload": payload,
            }
        )
    labels = records[0]["payload"]["labels"].tolist()
    if Counter(int(value) for value in labels) != Counter({class_id: 2 for class_id in range(100)}):
        raise AssertionError("Candidate pilot labels are not 2-per-class balanced")
    expected_labels, expected_references = common.load_frozen_pilot_references(
        parent["parent_dir"], pilot
    )
    if [int(value) for value in labels] != expected_labels:
        raise AssertionError("Candidate labels do not match the audited R3 official-test rows")
    observed_references = records[0]["payload"]["reference_top2"].tolist()
    if [[int(value) for value in row] for row in observed_references] != expected_references:
        raise AssertionError("Candidate references do not match the audited R3 standard rows")
    return records


def verify_reevaluation(
    protocol: Mapping[str, Any], output_root: Path, pilot_indices: Sequence[int]
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    directory = output_root / "common_mc1000"
    paths = {
        "rows": directory / "candidate_reeval_per_example.csv",
        "summaries": directory / "candidate_reeval_summary.csv",
        "families": directory / "attack_family_three_seed_summary.csv",
        "selected": directory / "selected_attack_envelope.csv",
        "comparisons": directory / "pilot_paired_bootstrap.csv",
        "freeze": directory / "confirmatory_attack_freeze.json",
        "final": directory / "final_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete MC=1000 reevaluation:\n  " + "\n  ".join(missing))
    rows = common.read_csv(paths["rows"])
    summaries = common.read_csv(paths["summaries"])
    families = common.read_csv(paths["families"])
    selected = common.read_csv(paths["selected"])
    comparisons = common.read_csv(paths["comparisons"])
    if len(rows) != 24 * 200 or len(summaries) != 24 or len(families) != 8 or len(selected) != 4:
        raise AssertionError("Unexpected R4 reevaluation table dimensions")
    expected_keys = {
        (method, attack, seed, index)
        for method, attack, seed in common.expected_cells()
        for index in range(200)
    }
    # Row positions are checked separately from dataset indices because frozen
    # sample indices are not contiguous.
    seen: set[tuple[str, str, int, int]] = set()
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], row["attack"], int(row["search_seed"]))
        groups.setdefault(key, []).append(row)
        position = len(groups[key]) - 1
        seen.add((*key, position))
        expected_draws = common.REEVAL_DRAWS if common.method_spec(row["method"]).stochastic else 1
        if int(row["eval_draws"]) != expected_draws or int(row["reeval_seed"]) != common.REEVAL_SEED:
            raise AssertionError("Reevaluation draw/seed mismatch")
        for metric in (
            "attacked_top1_accuracy_probability",
            "attacked_reference_top1_match_probability",
            "attacked_exact_set_probability",
            "attacked_label_coverage_probability",
            "clean_exact_set_probability",
        ):
            value = float(row[metric])
            if not 0 <= value <= 1:
                raise AssertionError(f"Probability outside [0,1]: {metric}")
    if seen != expected_keys or any(len(group) != 200 for group in groups.values()):
        raise AssertionError("R4 reevaluation row key coverage mismatch")
    reference_indices: list[int] | None = None
    for key, group in groups.items():
        indices = [int(row["sample_index"]) for row in group]
        if len(set(indices)) != 200:
            raise AssertionError(f"Duplicate reevaluation sample indices: {key}")
        if reference_indices is None:
            reference_indices = indices
        elif indices != reference_indices:
            raise AssertionError("Reevaluation cells do not use identical sample order")
    if reference_indices != [int(value) for value in pilot_indices]:
        raise AssertionError("Reevaluation rows do not follow the frozen R3 pilot order")
    summary_map = {
        (row["method"], row["attack"], int(row["search_seed"])): row
        for row in summaries
    }
    if set(summary_map) != set(groups):
        raise AssertionError("Reevaluation summary keys differ from per-example rows")
    for key, group in groups.items():
        observed = 100.0 * float(np.mean([float(row["attacked_exact_set_probability"]) for row in group]))
        assert_close(
            float(summary_map[key]["attacked_exact_set_probability_percent"]),
            observed,
            1e-9,
            f"Aggregate exact-set mismatch {key}",
        )
    expected_family, expected_selected = family_and_selection(summaries)
    family_map = {(row["method"], row["attack"]): row for row in families}
    for expected in expected_family:
        row = family_map.get((str(expected["method"]), str(expected["attack"])))
        if row is None:
            raise AssertionError("Missing attack-family row")
        for key in ("three_seed_mean_exact_set_percent", "three_seed_sd_pp"):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"Family summary mismatch {key}")
    selected_map = {row["method"]: row for row in selected}
    for expected in expected_selected:
        row = selected_map.get(str(expected["method"]))
        if row is None or row["selected_attack"] != expected["selected_attack"]:
            raise AssertionError("Selected attack family mismatch")
        if as_bool(row["paper_claim_authorized"]):
            raise AssertionError("Pilot selection incorrectly authorizes a paper claim")
    expected_comparisons = pilot_comparisons(rows, selected)
    saved_comparison_map = {row["comparison"]: row for row in comparisons}
    for expected in expected_comparisons:
        row = saved_comparison_map.get(str(expected["comparison"]))
        if row is None:
            raise AssertionError("Missing pilot comparison")
        for key in ("difference_pp", "ci95_low_pp", "ci95_high_pp"):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"Comparison mismatch {key}")
    freeze = common.read_json(paths["freeze"])
    claimed = str(freeze.get("selection_hash", ""))
    base = dict(freeze)
    base.pop("selection_hash", None)
    if common.object_sha256(base) != claimed:
        raise AssertionError("Confirmatory attack freeze hash mismatch")
    if freeze.get("protocol_hash") != protocol["protocol_hash"]:
        raise AssertionError("Confirmatory freeze protocol mismatch")
    if freeze.get("confirmatory_partition", {}).get("indices_sha256") != protocol[
        "confirmatory_indices_sha256"
    ]:
        raise AssertionError("Confirmatory freeze split hash mismatch")
    if freeze.get("confirmatory_partition", {}).get("outcomes_read") is not False:
        raise AssertionError("Confirmatory outcomes were read during pilot")
    if freeze.get("confirmatory_search_seeds") != list(common.CONFIRMATORY_SEARCH_SEEDS):
        raise AssertionError("Confirmatory search seeds changed")
    frozen_selection = freeze.get("method_attack_selection", {})
    for row in selected:
        if frozen_selection.get(row["method"], {}).get("attack") != row["selected_attack"]:
            raise AssertionError("Confirmatory freeze differs from selected attack table")
    final = common.read_json(paths["final"])
    if final.get("status") != "PASS" or final.get("confirmatory_attack_freeze_hash") != claimed:
        raise AssertionError("R4 final summary mismatch")
    if final.get("paper_claim_status") != "pilot_is_exploratory_only":
        raise AssertionError("R4 final summary overstates the pilot")
    return summaries, families, selected


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    parent_dir = args.parent_dir.expanduser().resolve()
    protocol, parent = verify_protocol(output_root, parent_dir)
    records = verify_search_cells(protocol, parent, output_root)
    summaries, families, selected = verify_reevaluation(
        protocol, output_root, parent["pilot_indices"]
    )
    print(
        f"PASS {output_root}: {len(records)} search cells, "
        f"{len(summaries)} MC reevaluations, candidates feasible",
        flush=True,
    )
    print("Selected pilot attack families (common MC=1000)", flush=True)
    for row in selected:
        print(
            f"  {row['method']:>11}: {row['selected_attack']} "
            f"Exact={float(row['selected_three_seed_mean_exact_set_percent']):.3f}% "
            f"sd={float(row['selected_three_seed_sd_pp']):.3f} pp",
            flush=True,
        )
    print("PASS confirmatory 1,000-example split remained outcome-blind", flush=True)
    print("CONFIRMATORY_ATTACK_STAGE_READY=YES", flush=True)
    print("R4 pilot audit complete; no merge, deletion, retuning, or per-example selection was performed.", flush=True)


if __name__ == "__main__":
    main()
