#!/usr/bin/env python3
"""Read-only audit of the ImageNet ViT-reference attack pilot VR3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import topk_imagenet_vit_reference_attack_pilot_vr3_common as common
import topk_imagenet_vit_reference_square_attack_core_vr3 as core
from prepare_topk_imagenet_vit_reference_attack_pilot_vr3 import validate_input_payload
from search_topk_imagenet_vit_reference_attack_pilot_vr3 import identity, validate_candidate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=common.DEFAULT_R0_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=common.DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--clean-dir", type=Path, default=common.DEFAULT_CLEAN_DIR)
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def expected_plan(method: str, attack: str) -> Mapping[str, Any]:
    stochastic = common.method_spec(method).stochastic
    if attack == "SquareAdaptive":
        cfg = common.ADAPTIVE_CONFIG
        return core.square_adaptive_query_plan(
            common.API_BUDGET, cfg["search_eot"], cfg["refresh_eot"],
            cfg["confirm_candidates"], cfg["confirm_fraction"], stochastic,
        )
    cfg = common.RACING_CONFIG
    if stochastic:
        values = (cfg["search_eot"], cfg["refresh_eot"], cfg["pool_size"], cfg["race_interval"], cfg["race_eot"])
    else:
        values = (1, 0, 1, common.API_BUDGET + 1, 0)
    return core.square_racing_query_plan(
        common.API_BUDGET, *values, cfg["confirm_fraction"], stochastic
    )


def validate_protocol_and_pilot(root: Path, parent: Mapping[str, Any]):
    protocol_path = root / "attack_pilot_protocol.json"
    protocol = common.read_json(protocol_path)
    protocol_hash = common.validate_protocol(protocol)
    if Path(protocol["paths"]["output_root"]).resolve() != root:
        raise AssertionError("Protocol output root differs from audited root")
    if protocol["pilot_rows_sha256"] != common.path_rows_sha256(common.select_pilot_rows(parent["split_rows"])):
        raise AssertionError("Outcome-blind pilot selection changed")
    if protocol["confirmatory_rows_sha256"] != common.path_rows_sha256(parent["confirmatory"]):
        raise AssertionError("Confirmatory partition identity changed")
    for name, expected in protocol["script_sha256"].items():
        path = common.PROJECT_DIR / name
        if not path.is_file() or common.file_sha256(path) != expected:
            raise AssertionError(f"VR3 source changed after protocol freeze: {name}")
    manifest = common.read_csv(root / "attack_pilot_manifest.csv")
    selected = common.select_pilot_rows(parent["split_rows"])
    if len(manifest) != common.PILOT_EXAMPLES:
        raise AssertionError("Pilot manifest row count changed")
    if any(row.get("protocol_hash") != protocol_hash for row in manifest):
        raise AssertionError("Pilot manifest protocol identity changed")
    if [row["relative_path"] for row in manifest] != [row["relative_path"] for row in selected]:
        raise AssertionError("Pilot manifest order differs from SHA-ranked selection")
    pilot_path = root / "pilot_inputs.pt"
    pilot = torch.load(pilot_path, map_location="cpu", weights_only=False)
    validate_input_payload(pilot, protocol)
    if list(pilot["relative_paths"]) != [row["relative_path"] for row in manifest]:
        raise AssertionError("Pilot tensor payload path order differs from manifest")
    if pilot["labels"].tolist() != [int(row["target"]) for row in manifest]:
        raise AssertionError("Pilot tensor labels differ from manifest")
    if list(pilot["class_names"]) != [row["class_name"] for row in manifest]:
        raise AssertionError("Pilot tensor class names differ from manifest")
    prepare = common.read_json(root / "prepare_summary.json")
    if prepare.get("status") != "PASS" or prepare.get("pilot_inputs_payload_hash") != pilot["payload_hash"]:
        raise AssertionError("Pilot preparation summary mismatch")
    if prepare.get("pilot_inputs_file_sha256") != common.file_sha256(pilot_path):
        raise AssertionError("Pilot input file changed")
    if prepare.get("pilot_manifest_file_sha256") != common.file_sha256(root / "attack_pilot_manifest.csv"):
        raise AssertionError("Pilot manifest file changed")
    return protocol, protocol_hash, pilot


def validate_search_cells(root: Path, protocol: Mapping[str, Any], pilot: Mapping[str, Any]):
    positions = [int(value) for value in pilot["pilot_positions"]]
    paths = list(pilot["relative_paths"])
    clean_hash = common.tensor_sha256(pilot["clean"])
    for method, attack, seed in common.expected_cells():
        directory = common.cell_dir(root, method, attack, seed)
        candidate_path, summary_path = directory / "candidate.pt", directory / "search_summary.json"
        if not candidate_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete search cell: {directory}")
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        expected_identity = identity(protocol, method, attack, seed, str(pilot["payload_hash"]))
        validate_candidate_payload(payload, expected_identity, positions, paths)
        if payload["tensor_sha256"]["clean"] != clean_hash:
            raise AssertionError("Candidate does not reuse the frozen clean tensor")
        details = payload["query_details"]
        plan = expected_plan(method, attack)
        for key in ("search_steps", "confirmation_queries", "logical_calls"):
            if int(details[key]) != int(plan[key]):
                raise AssertionError(f"Query-plan mismatch {method}/{attack}/{seed}: {key}")
        if int(details["api_queries"]) != common.API_BUDGET:
            raise AssertionError("A search cell did not consume exactly Q=5000")
        summary = common.read_json(summary_path)
        if summary.get("status") != "PASS":
            raise AssertionError("Search cell did not finish with PASS")
        if summary.get("candidate_payload_hash") != payload["payload_hash"]:
            raise AssertionError("Search summary/candidate payload mismatch")
        if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
            raise AssertionError("Search candidate file hash mismatch")
        linf = 255.0 * (payload["adversarial"] - payload["clean"]).abs().flatten(1).max(1).values
        if float(linf.max()) > common.EPSILON_PX + 2e-4:
            raise AssertionError("Candidate exceeds the frozen threat radius")
        print(f"PASS search {method}/{attack}/seed={seed}: Q=5000, candidates feasible", flush=True)


def validate_common_reevaluation(root: Path, protocol: Mapping[str, Any]):
    directory = root / "common_mc1000"
    required_counts = {
        "candidate_reeval_summary.csv": 24,
        "candidate_reeval_per_example.csv": 2400,
        "attack_family_three_seed_summary.csv": 8,
        "attack_family_paired_bootstrap.csv": 4,
        "selected_attack_envelope.csv": 4,
        "pilot_exploratory_comparisons.csv": 5,
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for name, count in required_counts.items():
        rows = common.read_csv(directory / name)
        if len(rows) != count:
            raise AssertionError(f"{name}: expected {count} rows, got {len(rows)}")
        tables[name] = rows
    summaries = tables["candidate_reeval_summary.csv"]
    per_example = tables["candidate_reeval_per_example.csv"]
    keys = {(row["method"], row["attack"], int(row["search_seed"])) for row in summaries}
    if keys != set(common.expected_cells()):
        raise AssertionError("Reevaluation summary cell set is incomplete")
    for row in summaries:
        method, attack, seed = row["method"], row["attack"], int(row["search_seed"])
        subset = [item for item in per_example if item["method"] == method and item["attack"] == attack and int(item["search_seed"]) == seed]
        if len(subset) != common.PILOT_EXAMPLES:
            raise AssertionError("Per-example reevaluation cell is incomplete")
        if sorted(int(item["pilot_position"]) for item in subset) != list(range(common.PILOT_EXAMPLES)):
            raise AssertionError("Per-example pilot positions changed")
        for metric in ("exact_set_k5_probability", "exact_set_k2_probability", "top1_accuracy_probability", "label_coverage_k5_probability"):
            observed = 100.0 * float(np.mean([float(item[f"attacked_{metric}"]) for item in subset]))
            if not close(observed, float(row[f"attacked_{metric}_percent"])):
                raise AssertionError(f"Aggregate does not reproduce rows: {method}/{attack}/{seed}/{metric}")
        expected_draws = common.REEVAL_DRAWS if common.method_spec(method).stochastic else 1
        if int(row["eval_draws"]) != expected_draws:
            raise AssertionError("Reevaluation draw count changed")
    selected = tables["selected_attack_envelope.csv"]
    selected_map = {row["method"]: row["selected_attack"] for row in selected}
    if set(selected_map) != set(common.METHODS) or not set(selected_map.values()) <= set(common.ATTACKS):
        raise AssertionError("Selected attack map is invalid")
    families = tables["attack_family_three_seed_summary.csv"]
    for method in common.METHODS:
        candidates = [row for row in families if row["method"] == method]
        candidates.sort(key=lambda row: (float(row["three_seed_mean_exact_set_k5_percent"]), 0 if row["attack"] == "SquareAdaptive" else 1))
        if selected_map[method] != candidates[0]["attack"]:
            raise AssertionError(f"Frozen selection rule not followed for {method}")
        selected_row = next(row for row in selected if row["method"] == method)
        if not close(selected_row["selected_three_seed_mean_exact_set_k5_percent"], candidates[0]["three_seed_mean_exact_set_k5_percent"]):
            raise AssertionError("Selected family aggregate mismatch")
    freeze = common.read_json(directory / "confirmatory_attack_freeze.json")
    claimed = freeze.get("selection_hash")
    base = dict(freeze)
    base.pop("selection_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError("Confirmatory attack freeze self-hash mismatch")
    if freeze.get("paper_claim_authorized_by_pilot") is not False:
        raise AssertionError("Pilot improperly authorized a paper claim")
    if freeze.get("confirmatory_partition", {}).get("outcomes_read") is not False:
        raise AssertionError("Confirmatory outcomes were not kept blinded")
    if freeze.get("confirmatory_partition", {}).get("rows_sha256") != protocol["confirmatory_rows_sha256"]:
        raise AssertionError("Confirmatory partition hash mismatch")
    final = common.read_json(directory / "final_summary.json")
    if final.get("status") != "PASS" or final.get("confirmatory_attack_freeze_hash") != claimed:
        raise AssertionError("Final VR3 summary/freeze mismatch")
    if final.get("confirmatory_attack_stage_ready") is not True:
        raise AssertionError("Final VR3 summary did not authorize confirmatory attack")
    if final.get("adversarial_paper_claim_authorized") is not False:
        raise AssertionError("Pilot improperly authorized an adversarial paper claim")
    if final.get("robust_vit_claim_authorized") is not False:
        raise AssertionError("Cross-architecture pilot improperly authorized a robust-ViT claim")
    return selected, families


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    parent = common.validate_parent(args.r0_dir, args.calibration_dir, args.clean_dir)
    protocol, _protocol_hash, pilot = validate_protocol_and_pilot(root, parent)
    validate_search_cells(root, protocol, pilot)
    selected, _families = validate_common_reevaluation(root, protocol)
    print("\nPASS ImageNet ViT-reference VR3 attack-family pilot", flush=True)
    for row in selected:
        print(f"  {row['method']:>11}: {row['selected_attack']} "
              f"Exact5={float(row['selected_three_seed_mean_exact_set_k5_percent']):.3f}% "
              f"sd={float(row['selected_three_seed_sd_pp']):.3f} pp", flush=True)
    print("PASS 24 exact-budget search cells, 2,400 common-MC rows, feasible candidates", flush=True)
    print("PASS confirmatory 1,000-image outcomes remain unread", flush=True)
    print("CONFIRMATORY_ATTACK_STAGE_READY=YES", flush=True)
    print("PAPER_CLAIM_AUTHORIZED=NO (pilot selection only)", flush=True)
    print("VR3 audit complete; no training, retuning, merge, deletion, or per-example selection was performed.", flush=True)


if __name__ == "__main__":
    main()
