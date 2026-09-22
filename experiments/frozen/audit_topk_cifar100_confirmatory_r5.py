#!/usr/bin/env python3
"""Independent structural/statistical audit for CIFAR-100 confirmatory R5."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import finalize_topk_cifar100_confirmatory_r5 as finalize
import topk_cifar100_confirmatory_r5_common as common
import topk_cifar100_square_attack_core_r4 as core
from search_topk_cifar100_confirmatory_r5 import payload_identity, validate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CIFAR-100 confirmatory R5")
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parent-dir", type=Path, default=common.DEFAULT_PARENT_DIR)
    parser.add_argument("--r4-root", type=Path, default=common.DEFAULT_R4_ROOT)
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
    output_root: Path, parent_dir: Path, r4_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = output_root / "confirmatory_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing frozen R5 protocol: {protocol_path}")
    protocol = common.read_json(protocol_path)
    common.validate_protocol(protocol)
    if Path(protocol["output_root"]).resolve() != output_root.resolve():
        raise AssertionError("R5 protocol output root differs from audit target")
    if Path(protocol["parent_dir"]).resolve() != parent_dir.resolve():
        raise AssertionError("R5 protocol parent directory differs from audit target")
    if Path(protocol["r4_root"]).resolve() != r4_root.resolve():
        raise AssertionError("R5 protocol R4 root differs from audit target")
    for name in common.SCRIPT_NAMES:
        path = common.PROJECT_DIR / name
        if not path.is_file() or common.file_sha256(path) != protocol["script_sha256"].get(name):
            raise AssertionError(f"Frozen R5 source changed: {name}")
    for name in common.DEPENDENCY_NAMES:
        path = common.PROJECT_DIR / name
        if not path.is_file() or common.file_sha256(path) != protocol["dependency_sha256"].get(name):
            raise AssertionError(f"Frozen R5 dependency changed: {name}")
    parent = common.load_parent_authorization(parent_dir)
    if parent["handoff_hash"] != protocol["parent_handoff_hash"]:
        raise AssertionError("R3 parent handoff changed after R5 freeze")
    if common.sequence_sha256(parent["confirmatory_indices"]) != protocol["indices_sha256"]:
        raise AssertionError("R5 confirmatory split differs from the R3 parent")
    if set(parent["confirmatory_indices"]) & set(parent["pilot_indices"]):
        raise AssertionError("R5 confirmatory split overlaps the R4 pilot")
    r4_info = common.load_r4_freeze(r4_root, parent)
    if r4_info["protocol_hash"] != protocol["r4_protocol_hash"]:
        raise AssertionError("R4 protocol changed after R5 freeze")
    if r4_info["selection_hash"] != protocol["r4_selection_hash"]:
        raise AssertionError("R4 attack-family selection changed after R5 freeze")
    return protocol, parent, r4_info


def expected_query_details(method: str) -> Mapping[str, Any]:
    attack = common.METHOD_ATTACK[method]
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
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    confirmatory = [int(value) for value in parent["confirmatory_indices"]]
    pilot = set(int(value) for value in parent["pilot_indices"])
    expected_labels, expected_references = common.load_frozen_confirmatory_references(
        parent["parent_dir"], confirmatory
    )
    common.validate_confirmatory_labels(expected_labels)
    records: list[dict[str, Any]] = []
    shared: dict[str, str] | None = None
    for method, attack, seed in common.expected_cells():
        directory = common.cell_dir(output_root, method, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        rows_path = directory / "search_per_example.csv"
        missing = [str(path) for path in (candidate_path, summary_path, rows_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete R5 search cell:\n  " + "\n  ".join(missing))
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        identity = payload_identity(
            protocol_hash=str(protocol["protocol_hash"]),
            method=method,
            search_seed=seed,
        )
        validate_payload(payload, identity, confirmatory)
        if set(int(value) for value in payload["sample_indices"].tolist()) & pilot:
            raise AssertionError("R5 candidate contains R4 pilot examples")
        if payload.get("query_details") != expected_query_details(method):
            raise AssertionError(f"Unexpected R5 query plan: {method}/{attack}/{seed}")
        if payload.get("r4_selection_hash") != common.EXPECTED_R4_SELECTION_HASH:
            raise AssertionError("R5 candidate attack-family freeze mismatch")
        summary = common.read_json(summary_path)
        if summary.get("status") != "PASS" or int(summary.get("examples", -1)) != 1000:
            raise AssertionError(f"Invalid R5 search summary: {directory}")
        if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
            raise AssertionError(f"R5 candidate file hash mismatch: {directory}")
        if summary.get("confirmatory_partition_read") is not True:
            raise AssertionError("R5 search summary does not identify confirmatory execution")
        if summary.get("pilot_partition_read") is not False:
            raise AssertionError("R5 search summary reports pilot outcome access")
        rows = common.read_csv(rows_path)
        if len(rows) != 1000:
            raise AssertionError(f"R5 search rows count mismatch: {directory}")
        if [int(row["sample_index"]) for row in rows] != confirmatory:
            raise AssertionError(f"R5 search row order mismatch: {directory}")
        if any(int(row["api_queries"]) != common.API_BUDGET for row in rows):
            raise AssertionError("R5 per-example query count differs from exact Q=5000")
        hashes = payload["tensor_sha256"]
        anchors = {
            key: hashes[key]
            for key in ("clean", "labels", "sample_indices", "reference_top2")
        }
        if shared is None:
            shared = anchors
        elif anchors != shared:
            raise AssertionError("R5 search cells do not share clean anchors")
        records.append({"method": method, "attack": attack, "seed": seed, "payload": payload})
    first = records[0]["payload"]
    labels = [int(value) for value in first["labels"].tolist()]
    common.validate_confirmatory_labels(labels)
    if labels != expected_labels:
        raise AssertionError("R5 labels differ from audited R3 rows")
    references = [[int(value) for value in row] for row in first["reference_top2"].tolist()]
    if references != expected_references:
        raise AssertionError("R5 references differ from audited R3 rows")
    return records


def _rows_by_key(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[str, ...], Mapping[str, Any]]:
    result: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        if key in result:
            raise AssertionError(f"Duplicate aggregate key: {key}")
        result[key] = row
    return result


def verify_analysis(protocol: Mapping[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    all_rows: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, _ = finalize.validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
    if len(all_rows) != 12000:
        raise AssertionError("R5 must contain exactly 12,000 reevaluation rows")
    finalize.validate_cross_method_alignment(all_rows)
    expected_method = finalize.build_method_summary(all_rows)
    expected_comparisons = finalize.build_comparisons(all_rows)
    expected_classwise = finalize.build_classwise(all_rows)

    directory = output_root / "confirmatory_analysis"
    paths = {
        "per_example": directory / "confirmatory_per_example.csv",
        "candidate_summary": directory / "confirmatory_candidate_summary.csv",
        "method_summary": directory / "confirmatory_method_summary.csv",
        "comparisons": directory / "confirmatory_paired_bootstrap.csv",
        "classwise": directory / "confirmatory_classwise.csv",
        "handoff": directory / "paper_claim_handoff.json",
        "final": directory / "final_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete R5 confirmatory analysis:\n  " + "\n  ".join(missing))
    saved_rows = common.read_csv(paths["per_example"])
    if saved_rows != all_rows:
        raise AssertionError("R5 consolidated per-example table differs from method tables")
    if len(common.read_csv(paths["candidate_summary"])) != 12:
        raise AssertionError("R5 candidate summary must contain 12 rows")
    saved_method = common.read_csv(paths["method_summary"])
    saved_comparisons = common.read_csv(paths["comparisons"])
    saved_classwise = common.read_csv(paths["classwise"])
    if len(saved_method) != 4 or len(saved_comparisons) != 16 or len(saved_classwise) != 400:
        raise AssertionError("R5 final table dimensions changed")

    observed_methods = _rows_by_key(saved_method, ("method",))
    for expected in expected_method:
        row = observed_methods.get((str(expected["method"]),))
        if row is None or row["attack"] != expected["attack"]:
            raise AssertionError("R5 method summary identity mismatch")
        for key in (
            "attacked_exact_set_probability_percent",
            "attacked_exact_set_seed_sd_pp",
            "clean_exact_set_probability_percent",
            "clean_to_attacked_exact_set_delta_pp",
            "attacked_top1_accuracy_probability_percent",
            "clean_top1_accuracy_probability_percent",
        ):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"R5 method summary {key}")
        if as_bool(row["seed_selection_performed"]):
            raise AssertionError("R5 method table reports seed selection")

    observed_comparisons = _rows_by_key(saved_comparisons, ("comparison",))
    for expected in expected_comparisons:
        row = observed_comparisons.get((str(expected["comparison"]),))
        if row is None or row["family"] != expected["family"]:
            raise AssertionError("R5 comparison identity mismatch")
        for key in ("difference_pp", "ci95_low_pp", "ci95_high_pp"):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"R5 comparison {key}")
        if as_bool(row["paper_claim_authorized"]) != bool(expected["paper_claim_authorized"]):
            raise AssertionError("R5 comparison authorization mismatch")

    observed_classwise = _rows_by_key(saved_classwise, ("method", "class_id"))
    for expected in expected_classwise:
        row = observed_classwise.get((str(expected["method"]), str(expected["class_id"])))
        if row is None:
            raise AssertionError("R5 classwise cell missing")
        for key in (
            "clean_exact_set_probability_percent",
            "attacked_exact_set_probability_percent",
            "clean_to_attacked_exact_set_delta_pp",
        ):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"R5 classwise {key}")

    handoff = common.read_json(paths["handoff"])
    handoff_hash = str(handoff.get("handoff_hash", ""))
    handoff_base = dict(handoff)
    handoff_base.pop("handoff_hash", None)
    if not handoff_hash or common.object_sha256(handoff_base) != handoff_hash:
        raise AssertionError("R5 paper-claim handoff self-hash mismatch")
    primary = next(row for row in expected_comparisons if row["comparison"] == common.PRIMARY_COMPARISON)
    primary_pass = primary["status"] == "positive"
    if handoff.get("authorized") is not primary_pass:
        raise AssertionError("R5 paper-claim handoff decision mismatch")
    if handoff.get("no_attack_family_selection_on_confirmatory_data") is not True:
        raise AssertionError("R5 handoff permits confirmatory attack selection")

    final = common.read_json(paths["final"])
    claimed = str(final.get("final_summary_hash", ""))
    base = dict(final)
    base.pop("final_summary_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError("R5 final summary self-hash mismatch")
    if final.get("protocol_hash") != protocol["protocol_hash"]:
        raise AssertionError("R5 final/protocol identity mismatch")
    if final.get("paper_claim_authorized") is not primary_pass:
        raise AssertionError("R5 final paper-claim decision mismatch")
    if final.get("paper_claim_handoff_hash") != handoff_hash:
        raise AssertionError("R5 final/handoff identity mismatch")
    if any(
        final.get(key) is not False
        for key in (
            "confirmatory_attack_selection_performed",
            "confirmatory_seed_selection_performed",
            "defense_retuning_performed",
        )
    ):
        raise AssertionError("R5 final summary reports forbidden tuning or selection")
    for name, expected_hash in final.get("files", {}).items():
        path = directory / name
        if not path.is_file() or common.file_sha256(path) != expected_hash:
            raise AssertionError(f"R5 final file hash mismatch: {name}")
    return expected_method, primary


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    parent_dir = args.parent_dir.expanduser().resolve()
    r4_root = args.r4_root.expanduser().resolve()
    protocol, parent, _ = verify_protocol(output_root, parent_dir, r4_root)
    records = verify_search_cells(protocol, parent, output_root)
    method_summary, primary = verify_analysis(protocol, output_root)
    print(
        f"PASS {output_root}: {len(records)} frozen attack cells, "
        "12,000 common-evaluation rows, candidates feasible",
        flush=True,
    )
    print("CIFAR-100 R5 confirmatory summary", flush=True)
    for row in method_summary:
        print(
            f"  {row['method']:>11}: {row['attack']} "
            f"Exact={float(row['attacked_exact_set_probability_percent']):.3f}% "
            f"sd={float(row['attacked_exact_set_seed_sd_pp']):.3f} pp",
            flush=True,
        )
    print(
        f"  PRIMARY Full-Robust={float(primary['difference_pp']):+.3f} pp "
        f"[{float(primary['ci95_low_pp']):+.3f}, "
        f"{float(primary['ci95_high_pp']):+.3f}] -> {primary['status']}",
        flush=True,
    )
    print("PASS attacks were frozen on the disjoint R4 pilot; no confirmatory selection occurred", flush=True)
    print(
        f"PAPER_CLAIM_AUTHORIZED={'YES' if primary['status'] == 'positive' else 'NO'}",
        flush=True,
    )
    print("R5 confirmatory audit complete; no training, retuning, merge, deletion, or selection was performed.", flush=True)


if __name__ == "__main__":
    main()
