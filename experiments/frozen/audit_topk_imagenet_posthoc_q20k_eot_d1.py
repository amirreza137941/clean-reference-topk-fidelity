#!/usr/bin/env python3
"""Independent structural/statistical audit for the D1 post-hoc diagnostic."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import finalize_topk_imagenet_posthoc_q20k_eot_d1 as finalize
import topk_cifar100_square_attack_core_r4 as core
import topk_imagenet_posthoc_q20k_eot_d1_common as common
from prepare_topk_imagenet_posthoc_q20k_eot_d1 import validate_input_payload
from search_topk_imagenet_posthoc_q20k_eot_d1 import (
    candidate_identity,
    validate_candidate_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def assert_close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(float(observed)) or abs(float(observed) - expected) > tolerance:
        raise AssertionError(
            f"{label}: observed={observed}, expected={expected}, tol={tolerance}"
        )


def string_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{key: str(value) for key, value in row.items()} for row in rows]


def expected_query_details(method: str) -> dict[str, Any]:
    config = common.ADAPTIVE_CONFIG_BY_METHOD[method]
    plan = core.square_adaptive_query_plan(
        common.API_BUDGET,
        int(config["search_eot"]),
        int(config["refresh_eot"]),
        int(config["confirm_candidates"]),
        float(config["confirm_fraction"]),
        common.method_spec(method).stochastic,
    )
    return {
        "search_steps": plan["search_steps"],
        "search_eot": config["search_eot"],
        "refresh_eot": config["refresh_eot"],
        "confirmation_candidates": config["confirm_candidates"],
        "confirmation_queries": plan["confirmation_queries"],
        "confirmation_draws_min": min(plan["confirmation_draws"]),
        "confirmation_draws_max": max(plan["confirmation_draws"]),
        "logical_calls": plan["logical_calls"],
        "api_queries": common.API_BUDGET,
        "score_objective": config["objective"],
    }


def verify_protocol_and_preparation(
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = output_root / "diagnostic_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing D1 protocol: {protocol_path}")
    protocol = common.read_json(protocol_path)
    common.validate_protocol(protocol)
    if Path(protocol["paths"]["output_root"]).expanduser().resolve() != output_root:
        raise AssertionError("D1 protocol output root changed")
    if Path(protocol["paths"]["r5_root"]).expanduser().resolve() != (
        common.DEFAULT_R5_ROOT.resolve()
    ):
        raise AssertionError("D1 parent R5 path changed")
    parent = common.load_parent_r5(Path(protocol["paths"]["r5_root"]))
    subset = parent["subset"]
    if common.subset_rows_sha256(subset) != protocol["subset_rows_sha256"]:
        raise AssertionError("D1 subset identity changed")

    manifest_path = output_root / "diagnostic_manifest.csv"
    input_path = output_root / "diagnostic_inputs.pt"
    prepare_path = output_root / "prepare_summary.json"
    missing = [
        str(path)
        for path in (manifest_path, input_path, prepare_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Incomplete D1 preparation:\n  " + "\n  ".join(missing))
    expected_manifest = string_rows(
        [
            {
                **row,
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": protocol["protocol_hash"],
            }
            for row in subset
        ]
    )
    if common.read_csv(manifest_path) != expected_manifest:
        raise AssertionError("D1 manifest differs from the frozen path-only subset")
    inputs = torch.load(input_path, map_location="cpu", weights_only=False)
    validate_input_payload(inputs, protocol, subset)
    prepare = common.read_json(prepare_path)
    expected_prepare = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "status": "PASS",
        "post_hoc": True,
        "paper_claim_authorized": False,
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "subset_rows_sha256": protocol["subset_rows_sha256"],
        "diagnostic_inputs_payload_hash": inputs["payload_hash"],
        "diagnostic_inputs_file_sha256": common.file_sha256(input_path),
        "diagnostic_manifest_file_sha256": common.file_sha256(manifest_path),
        "clean_tensor_sha256": inputs["tensor_sha256"]["clean"],
        "parent_r5_inputs_file_sha256": common.EXPECTED_R5_INPUTS_FILE_SHA256,
        "parent_r5_clean_tensor_sha256": common.EXPECTED_R5_CLEAN_TENSOR_SHA256,
        "parent_r5_full_clean_stream_verified": True,
        "parent_r5_unchanged": True,
    }
    if prepare != expected_prepare:
        raise AssertionError("D1 preparation summary changed")
    return protocol, parent, inputs


def verify_search_cells(
    protocol: Mapping[str, Any],
    inputs: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    clean = inputs["clean"].detach().cpu()
    for method, attack, seed in common.expected_cells():
        directory = common.cell_dir(output_root, method, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        rows_path = directory / "search_per_example.csv"
        missing = [
            str(path)
            for path in (candidate_path, summary_path, rows_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Incomplete D1 search cell:\n  " + "\n  ".join(missing)
            )
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        identity = candidate_identity(
            protocol,
            method,
            seed,
            str(inputs["payload_hash"]),
            str(inputs["tensor_sha256"]["clean"]),
        )
        validate_candidate_payload(
            payload,
            identity,
            list(range(common.DIAGNOSTIC_EXAMPLES)),
            inputs["source_confirmatory_positions"].tolist(),
            inputs["relative_paths"],
            inputs["labels"],
            inputs["reference_top5"],
            clean=clean,
        )
        expected_details = expected_query_details(method)
        if payload.get("query_details") != expected_details:
            raise AssertionError(f"Unexpected D1 query plan: {method}/{seed}")
        if int(payload.get("batches", -1)) != math.ceil(
            common.DIAGNOSTIC_EXAMPLES
            / common.SEARCH_BATCH_SIZE_BY_METHOD[method]
        ):
            raise AssertionError(f"D1 batch count changed: {method}/{seed}")
        summary = common.read_json(summary_path)
        expected_summary_identity = {
            **identity,
            "status": "PASS",
            "examples": common.DIAGNOSTIC_EXAMPLES,
            "candidate_schema": "posthoc_shared_clean_plus_adversarial_v1",
            "candidate_file_sha256": common.file_sha256(candidate_path),
            "candidate_payload_hash": payload["payload_hash"],
            "candidate_tensor_sha256": payload["tensor_sha256"]["adversarial"],
            "search_per_example_sha256": common.file_sha256(rows_path),
            "query_details": expected_details,
            "parent_r5_modified": False,
            "seed_selection_performed": False,
            "attack_family_selection_performed": False,
            "independent_mc1000_complete": False,
        }
        for key, expected in expected_summary_identity.items():
            if summary.get(key) != expected:
                raise AssertionError(f"D1 search summary changed: {method}/{seed}/{key}")
        if float(summary["maximum_linf_px"]) > common.EPSILON_PX + 2.55e-4:
            raise AssertionError(f"D1 search Linf exceeds 4/255: {method}/{seed}")

        search_rows = common.read_csv(rows_path)
        if len(search_rows) != common.DIAGNOSTIC_EXAMPLES:
            raise AssertionError(f"D1 search table row count changed: {method}/{seed}")
        for position, row in enumerate(search_rows):
            if (
                int(row["diagnostic_position"]) != position
                or int(row["source_confirmatory_position"])
                != int(inputs["source_confirmatory_positions"][position])
                or row["relative_path"] != inputs["relative_paths"][position]
                or int(row["label"]) != int(inputs["labels"][position])
                or int(row["api_queries"]) != common.API_BUDGET
                or int(row["search_eot"])
                != common.ADAPTIVE_CONFIG_BY_METHOD[method]["search_eot"]
                or int(row["refresh_eot"])
                != common.ADAPTIVE_CONFIG_BY_METHOD[method]["refresh_eot"]
            ):
                raise AssertionError(f"D1 search row changed: {method}/{seed}/{position}")
            if float(row["linf"]) > common.EPSILON_PX / 255.0 + 1e-6:
                raise AssertionError(f"D1 row Linf exceeds radius: {method}/{seed}")
        records.append(
            {
                "method": method,
                "attack": attack,
                "search_seed": seed,
                "candidate_file_sha256": common.file_sha256(candidate_path),
                "candidate_payload_hash": payload["payload_hash"],
            }
        )
    if len(records) != len(common.METHODS) * len(common.SEARCH_SEEDS):
        raise AssertionError("D1 search-cell coverage is incomplete")
    return records


def verify_analysis(
    protocol: Mapping[str, Any], parent: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    analysis = output_root / "diagnostic_analysis"
    required = {
        "diagnostic_per_example.csv": analysis / "diagnostic_per_example.csv",
        "diagnostic_candidate_summary.csv": analysis
        / "diagnostic_candidate_summary.csv",
        "diagnostic_method_summary.csv": analysis / "diagnostic_method_summary.csv",
        "diagnostic_q20k_eot_stress_comparison_per_example.csv": analysis
        / "diagnostic_q20k_eot_stress_comparison_per_example.csv",
        "diagnostic_paired_bootstrap.csv": analysis
        / "diagnostic_paired_bootstrap.csv",
        "final_summary.json": analysis / "final_summary.json",
        "claim_boundary.json": analysis / "claim_boundary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete D1 analysis:\n  " + "\n  ".join(missing))

    all_rows: list[dict[str, str]] = []
    all_summaries: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, summaries = finalize.validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
    finalize.validate_cross_method_alignment(all_rows)
    expected_methods = finalize.build_method_summary(all_rows)
    parent_rows = common.read_csv(parent["paths"]["per_example"])
    expected_budget, values = finalize.build_budget_rows(
        all_rows, parent_rows, parent["subset"]
    )
    expected_comparisons = finalize.build_comparisons(values)

    expected_tables = {
        "diagnostic_per_example.csv": string_rows(all_rows),
        "diagnostic_candidate_summary.csv": string_rows(all_summaries),
        "diagnostic_method_summary.csv": string_rows(expected_methods),
        "diagnostic_q20k_eot_stress_comparison_per_example.csv": string_rows(
            expected_budget
        ),
        "diagnostic_paired_bootstrap.csv": string_rows(expected_comparisons),
    }
    for name, expected in expected_tables.items():
        observed = common.read_csv(required[name])
        if observed != expected:
            raise AssertionError(f"D1 analysis table failed reproduction: {name}")

    final = common.read_json(required["final_summary.json"])
    claimed_hash = str(final.get("final_summary_hash", ""))
    base = dict(final)
    base.pop("final_summary_hash", None)
    if not claimed_hash or common.object_sha256(base) != claimed_hash:
        raise AssertionError("D1 final-summary self-hash mismatch")
    if (
        final.get("protocol_id") != common.PROTOCOL_ID
        or final.get("protocol_hash") != protocol["protocol_hash"]
        or final.get("status") != "PASS"
        or final.get("post_hoc") is not True
        or final.get("paper_claim_authorized") is not False
        or final.get("may_revise_parent_r5_claim") is not False
        or final.get("parent_r5_modified") is not False
        or final.get("parent_r5_paper_claim_remains_authorized") is not True
        or final.get("budget_and_eot_effects_separately_identifiable") is not False
        or final.get("r5_vs_d1_reevaluation_common_random_numbers") is not False
        or final.get("no_training") is not True
        or final.get("no_defense_retuning") is not True
        or final.get("no_attack_family_selection") is not True
        or final.get("no_seed_selection") is not True
    ):
        raise AssertionError("D1 final diagnostic/claim boundary changed")
    for name in expected_tables:
        if final.get("files", {}).get(name) != common.file_sha256(required[name]):
            raise AssertionError(f"D1 final file hash mismatch: {name}")
    primary = next(
        row for row in expected_comparisons if row["family"] == "primary_diagnostic"
    )
    if final.get("primary_diagnostic") != primary:
        raise AssertionError("D1 primary diagnostic failed reproduction")

    claim = common.read_json(required["claim_boundary.json"])
    expected_claim = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "diagnostic_complete": True,
        "post_hoc": True,
        "paper_claim_authorized": False,
        "may_revise_parent_r5_claim": False,
        "parent_r5_claim_status": "unchanged_and_authorized_by_R5_only",
        "parent_r5_protocol_hash": common.EXPECTED_R5_PROTOCOL_HASH,
        "parent_r5_final_summary_hash": common.EXPECTED_R5_FINAL_SUMMARY_HASH,
    }
    if claim != expected_claim:
        raise AssertionError("D1 claim-boundary artifact changed")
    return final


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    protocol, parent, inputs = verify_protocol_and_preparation(output_root)
    records = verify_search_cells(protocol, inputs, output_root)
    final = verify_analysis(protocol, parent, output_root)
    audit = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "status": "PASS",
        "search_cells": len(records),
        "common_mc_rows": common.DIAGNOSTIC_EXAMPLES
        * len(common.SEARCH_SEEDS)
        * len(common.METHODS),
        "api_budget_per_example": common.API_BUDGET,
        "full_search_eot": common.ADAPTIVE_CONFIG_BY_METHOD["full"]["search_eot"],
        "robust_search_eot": common.ADAPTIVE_CONFIG_BY_METHOD["robust"][
            "search_eot"
        ],
        "parent_r5_protocol_hash": common.EXPECTED_R5_PROTOCOL_HASH,
        "parent_r5_final_summary_hash": common.EXPECTED_R5_FINAL_SUMMARY_HASH,
        "parent_r5_modified": False,
        "post_hoc": True,
        "paper_claim_authorized": False,
        "may_revise_parent_r5_claim": False,
        "final_summary_hash": final["final_summary_hash"],
    }
    audit["audit_summary_hash"] = common.object_sha256(audit)
    common.atomic_json(
        output_root / "diagnostic_analysis" / "audit_summary.json", audit
    )
    print(
        f"PASS {output_root}: {len(records)} frozen D1 cells, "
        f"{audit['common_mc_rows']} reevaluation rows, exact Q={common.API_BUDGET}",
        flush=True,
    )
    print(
        "PASS EOT schedule: Robust=1, Full=5; all oracle/refresh/confirmation "
        "draws charged",
        flush=True,
    )
    print("PASS parent R5 artifacts are byte-anchored and unchanged", flush=True)
    print("PAPER_CLAIM_AUTHORIZED=NO (post-hoc diagnostic)", flush=True)
    print("R5_PAPER_CLAIM_STATUS=UNCHANGED", flush=True)
    print("ImageNet ResNet-50 D1 audit complete; no training or retuning.", flush=True)


if __name__ == "__main__":
    main()
