#!/usr/bin/env python3
"""Independent structural/statistical audit for ImageNet ViT-reference VR4."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import calibrate_topk_imagenet_vit_reference_vr1 as vr1
import finalize_topk_imagenet_vit_reference_confirmatory_vr4 as finalize
import topk_imagenet_vit_reference_confirmatory_vr4_common as common
import topk_imagenet_vit_reference_square_attack_core_vr3 as core
from prepare_topk_imagenet_vit_reference_confirmatory_vr4 import validate_input_payload
from search_topk_imagenet_vit_reference_confirmatory_vr4 import (
    candidate_identity,
    validate_candidate_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=common.DEFAULT_R0_DIR)
    parser.add_argument(
        "--calibration-dir", type=Path, default=common.DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument("--clean-dir", type=Path, default=common.DEFAULT_CLEAN_DIR)
    parser.add_argument("--vr3-root", type=Path, default=common.DEFAULT_VR3_ROOT)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def assert_close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: observed={observed}, expected={expected}")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def verify_protocol_and_preparation(
    output_root: Path,
    r0_dir: Path,
    calibration_dir: Path,
    clean_dir: Path,
    vr3_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol_path = output_root / "confirmatory_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing ImageNet ViT-reference VR4 protocol: {protocol_path}")
    protocol = common.read_json(protocol_path)
    common.validate_protocol(protocol)
    expected_paths = {
        "output_root": output_root.resolve(),
        "r0_dir": r0_dir.resolve(),
        "calibration_dir": calibration_dir.resolve(),
        "clean_dir": clean_dir.resolve(),
        "vr3_root": vr3_root.resolve(),
    }
    for key, expected in expected_paths.items():
        if Path(protocol["paths"][key]).expanduser().resolve() != expected:
            raise AssertionError(f"ImageNet ViT-reference VR4 protocol path mismatch: {key}")
    parent = common.load_parent(r0_dir, calibration_dir, clean_dir)
    vr3_info = common.load_vr3_freeze(vr3_root, parent)
    if vr3_info["protocol_hash"] != protocol["vr3_protocol_hash"]:
        raise AssertionError("ImageNet VR3 protocol changed after VR4 freeze")
    if vr3_info["selection_hash"] != protocol["vr3_selection_hash"]:
        raise AssertionError("ImageNet VR3 selection changed after VR4 freeze")
    rows = common.canonical_confirmatory_rows(parent)

    manifest_path = output_root / "confirmatory_manifest.csv"
    input_path = output_root / "confirmatory_inputs.pt"
    summary_path = output_root / "prepare_summary.json"
    missing = [str(path) for path in (manifest_path, input_path, summary_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete ImageNet ViT-reference VR4 preparation:\n  " + "\n  ".join(missing))
    manifest = common.read_csv(manifest_path)
    expected_manifest = [
        {
            **{key: str(value) for key, value in row.items()},
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol["protocol_hash"],
        }
        for row in rows
    ]
    if manifest != expected_manifest:
        raise AssertionError("ImageNet ViT-reference VR4 saved manifest differs from the frozen split")
    inputs = torch.load(input_path, map_location="cpu", weights_only=False)
    validate_input_payload(inputs, protocol, rows)
    summary = common.read_json(summary_path)
    if summary.get("status") != "PASS" or summary.get("confirmatory_outcomes_read") is not True:
        raise AssertionError("ImageNet ViT-reference VR4 preparation summary is invalid")
    if summary.get("attack_family_selection_performed") is not False:
        raise AssertionError("ImageNet ViT-reference VR4 preparation reports forbidden attack selection")
    if summary.get("defense_retuning_performed") is not False:
        raise AssertionError("ImageNet ViT-reference VR4 preparation reports forbidden defense retuning")
    if summary.get("confirmatory_inputs_payload_hash") != inputs["payload_hash"]:
        raise AssertionError("ImageNet ViT-reference VR4 preparation payload hash mismatch")
    if summary.get("confirmatory_inputs_file_sha256") != common.file_sha256(input_path):
        raise AssertionError("ImageNet ViT-reference VR4 preparation file hash mismatch")
    return protocol, parent, rows, inputs


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


def load_clean_images_for_audit(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
    num_workers: int,
) -> torch.Tensor:
    loader = vr1.make_loader(
        Path(protocol["paths"]["data_root"]),
        rows,
        common.REFERENCE_BATCH_SIZE,
        num_workers,
        torch.device("cpu"),
    )
    images_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    paths_all: list[str] = []
    for images, labels, paths in loader:
        images_all.append(images.detach().cpu().contiguous())
        labels_all.append(labels.detach().cpu().contiguous())
        paths_all.extend(str(path) for path in paths)
    images = torch.cat(images_all)
    labels = torch.cat(labels_all)
    if paths_all != list(inputs["relative_paths"]) or not torch.equal(labels, inputs["labels"]):
        raise AssertionError("ImageNet ViT-reference VR4 audit data no longer matches frozen inputs")
    if common.tensor_sha256(images) != inputs["clean_tensor_sha256"]:
        raise AssertionError("ImageNet ViT-reference VR4 clean image tensor changed before audit")
    return images


def verify_search_cells(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
    clean: torch.Tensor,
    output_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    positions = list(range(common.CONFIRMATORY_EXAMPLES))
    paths = list(inputs["relative_paths"])
    shared: dict[str, str] | None = None
    for method, attack, seed in common.expected_cells():
        directory = common.cell_dir(output_root, method, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        rows_path = directory / "search_per_example.csv"
        missing = [str(path) for path in (candidate_path, summary_path, rows_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete ImageNet ViT-reference VR4 search cell:\n  " + "\n  ".join(missing))
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        identity = candidate_identity(
            protocol,
            method,
            seed,
            str(inputs["payload_hash"]),
            str(inputs["clean_tensor_sha256"]),
        )
        validate_candidate_payload(
            payload,
            identity,
            positions,
            paths,
            inputs["labels"],
            inputs["reference_top5"],
            clean=clean,
        )
        if payload.get("query_details") != expected_query_details(method):
            raise AssertionError(f"Unexpected ImageNet ViT-reference VR4 query plan: {method}/{attack}/{seed}")
        if payload.get("vr3_selection_hash") != common.EXPECTED_VR3_SELECTION_HASH:
            raise AssertionError("ImageNet ViT-reference VR4 candidate attack-family freeze changed")
        summary = common.read_json(summary_path)
        if summary.get("status") != "PASS" or int(summary.get("examples", -1)) != 1000:
            raise AssertionError(f"Invalid ImageNet ViT-reference VR4 search summary: {directory}")
        if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
            raise AssertionError(f"ImageNet ViT-reference VR4 candidate file hash mismatch: {directory}")
        if summary.get("candidate_payload_hash") != payload["payload_hash"]:
            raise AssertionError(f"ImageNet ViT-reference VR4 candidate payload mismatch: {directory}")
        if summary.get("confirmatory_partition_read") is not True:
            raise AssertionError("ImageNet ViT-reference VR4 summary omits confirmatory execution")
        if summary.get("pilot_partition_read") is not False:
            raise AssertionError("ImageNet ViT-reference VR4 summary reports pilot-outcome access")
        for forbidden in (
            "attack_family_selection_performed",
            "seed_selection_performed",
            "paper_claim_authorized_by_search",
        ):
            if summary.get(forbidden) is not False:
                raise AssertionError(f"ImageNet ViT-reference VR4 search summary violates {forbidden}")
        saved_rows = common.read_csv(rows_path)
        if len(saved_rows) != common.CONFIRMATORY_EXAMPLES:
            raise AssertionError(f"ImageNet ViT-reference VR4 search row count changed: {directory}")
        if [int(row["confirmatory_position"]) for row in saved_rows] != positions:
            raise AssertionError(f"ImageNet ViT-reference VR4 search row order changed: {directory}")
        if [row["relative_path"] for row in saved_rows] != paths:
            raise AssertionError(f"ImageNet ViT-reference VR4 search paths changed: {directory}")
        if any(int(row["api_queries"]) != common.API_BUDGET for row in saved_rows):
            raise AssertionError("ImageNet ViT-reference VR4 per-example query count is not exact Q=5000")
        actual_linf = (payload["adversarial"] - clean).abs().flatten(1).max(1).values.numpy()
        saved_linf = np.asarray([float(row["linf"]) for row in saved_rows])
        if not np.allclose(actual_linf, saved_linf, atol=1e-12, rtol=0):
            raise AssertionError(f"ImageNet ViT-reference VR4 saved Linf values changed: {directory}")
        assert_close(float(summary["maximum_linf"]), float(actual_linf.max()), 1e-12, "maximum Linf")
        if float(actual_linf.max()) > common.EPSILON_PX / 255.0 + 1e-6:
            raise AssertionError(f"ImageNet ViT-reference VR4 candidate is infeasible: {directory}")
        anchors = {
            key: payload["tensor_sha256"][key]
            for key in ("labels", "confirmatory_positions", "reference_top5")
        }
        anchors["relative_paths_sha256"] = common.object_sha256(payload["relative_paths"])
        if shared is None:
            shared = anchors
        elif anchors != shared:
            raise AssertionError("ImageNet ViT-reference VR4 search cells do not share anchors")
        records.append({"method": method, "attack": attack, "seed": seed})
        del payload
    return records


def _rows_by_key(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    result: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        if key in result:
            raise AssertionError(f"Duplicate ImageNet ViT-reference VR4 aggregate key: {key}")
        result[key] = row
    return result


def verify_analysis(
    protocol: Mapping[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    all_rows: list[dict[str, str]] = []
    for method in common.METHODS:
        rows, _summaries = finalize.validate_method_artifacts(protocol, method)
        all_rows.extend(rows)
    if len(all_rows) != 12_000:
        raise AssertionError("ImageNet ViT-reference VR4 must contain exactly 12,000 reevaluation rows")
    finalize.validate_cross_method_alignment(all_rows)
    expected_methods = finalize.build_method_summary(all_rows)
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
        raise FileNotFoundError("Incomplete ImageNet ViT-reference VR4 analysis:\n  " + "\n  ".join(missing))
    if common.read_csv(paths["per_example"]) != all_rows:
        raise AssertionError("ImageNet ViT-reference VR4 consolidated rows differ from method rows")
    if len(common.read_csv(paths["candidate_summary"])) != 12:
        raise AssertionError("ImageNet ViT-reference VR4 candidate summary must contain 12 rows")
    saved_methods = common.read_csv(paths["method_summary"])
    saved_comparisons = common.read_csv(paths["comparisons"])
    saved_classwise = common.read_csv(paths["classwise"])
    if len(saved_methods) != 4 or len(saved_comparisons) != 17 or len(saved_classwise) != 4000:
        raise AssertionError("ImageNet ViT-reference VR4 final table dimensions changed")

    method_map = _rows_by_key(saved_methods, ("method",))
    for expected in expected_methods:
        row = method_map.get((str(expected["method"]),))
        if row is None or row["attack"] != expected["attack"]:
            raise AssertionError("ImageNet ViT-reference VR4 method summary identity changed")
        for key in (
            "attacked_exact_set_k5_probability_percent",
            "attacked_exact_set_k5_seed_sd_pp",
            "clean_exact_set_k5_probability_percent",
            "clean_to_attacked_exact_set_k5_delta_pp",
            "attacked_exact_set_k2_probability_percent",
            "attacked_top1_accuracy_probability_percent",
            "clean_top1_accuracy_probability_percent",
        ):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"method summary {key}")
        if as_bool(row["seed_selection_performed"]):
            raise AssertionError("ImageNet ViT-reference VR4 method table reports seed selection")

    comparison_map = _rows_by_key(saved_comparisons, ("comparison",))
    for expected in expected_comparisons:
        row = comparison_map.get((str(expected["comparison"]),))
        if row is None or row["family"] != expected["family"]:
            raise AssertionError("ImageNet ViT-reference VR4 comparison identity changed")
        for key in ("difference_pp", "ci95_low_pp", "ci95_high_pp"):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"comparison {key}")
        if as_bool(row["paper_claim_authorized"]) != bool(expected["paper_claim_authorized"]):
            raise AssertionError("ImageNet ViT-reference VR4 comparison authorization changed")

    classwise_map = _rows_by_key(saved_classwise, ("method", "class_id"))
    for expected in expected_classwise:
        row = classwise_map.get((str(expected["method"]), str(expected["class_id"])))
        if row is None:
            raise AssertionError("ImageNet ViT-reference VR4 classwise cell missing")
        for key in (
            "clean_exact_set_k5_probability_percent",
            "attacked_exact_set_k5_probability_percent",
            "clean_to_attacked_exact_set_k5_delta_pp",
        ):
            assert_close(float(row[key]), float(expected[key]), 1e-9, f"classwise {key}")

    handoff = common.read_json(paths["handoff"])
    handoff_hash = str(handoff.get("handoff_hash", ""))
    handoff_base = dict(handoff)
    handoff_base.pop("handoff_hash", None)
    if not handoff_hash or common.object_sha256(handoff_base) != handoff_hash:
        raise AssertionError("ImageNet ViT-reference VR4 paper-claim handoff hash mismatch")
    primary = next(
        row for row in expected_comparisons if row["comparison"] == common.PRIMARY_COMPARISON
    )
    primary_pass = primary["status"] == "positive"
    if handoff.get("authorized") is not primary_pass:
        raise AssertionError("ImageNet ViT-reference VR4 handoff decision changed")
    if handoff.get("claim_scope") != "cross_architecture_vit_reference_transfer":
        raise AssertionError("ImageNet ViT-reference VR4 handoff claim scope changed")
    if handoff.get("architecture_roles") != protocol["architecture_roles"]:
        raise AssertionError("ImageNet ViT-reference VR4 handoff architecture roles changed")
    if handoff.get("robust_vit_claim_authorized") is not False:
        raise AssertionError("ImageNet ViT-reference VR4 handoff permits a robust-ViT claim")
    expected_scope = (
        protocol["analysis_plan"]["primary_claim_scope"] if primary_pass else None
    )
    if handoff.get("authorization_scope") != expected_scope:
        raise AssertionError("ImageNet ViT-reference VR4 handoff authorization scope changed")
    claim_text = handoff.get("claim_text_if_authorized")
    if primary_pass:
        if not isinstance(claim_text, str) or not all(
            phrase in claim_text
            for phrase in (
                "robust ResNet-50 defended backbone",
                "standard ViT-B/16 Top-5 reference",
                "does not constitute a robust-ViT claim",
            )
        ):
            raise AssertionError("ImageNet ViT-reference VR4 authorized claim text is unsafe")
    elif claim_text is not None:
        raise AssertionError("ImageNet ViT-reference VR4 unauthorized claim text is non-null")
    if handoff.get("no_attack_family_selection_on_confirmatory_data") is not True:
        raise AssertionError("ImageNet ViT-reference VR4 handoff permits confirmatory attack selection")
    if handoff.get("no_seed_selection") is not True:
        raise AssertionError("ImageNet ViT-reference VR4 handoff permits seed selection")
    if handoff.get("no_per_example_candidate_selection") is not True:
        raise AssertionError("ImageNet ViT-reference VR4 handoff permits per-example selection")

    final = common.read_json(paths["final"])
    claimed = str(final.get("final_summary_hash", ""))
    base = dict(final)
    base.pop("final_summary_hash", None)
    if not claimed or common.object_sha256(base) != claimed:
        raise AssertionError("ImageNet ViT-reference VR4 final-summary hash mismatch")
    if final.get("status") != "PASS":
        raise AssertionError("ImageNet ViT-reference VR4 final status is not PASS")
    if final.get("protocol_hash") != protocol["protocol_hash"]:
        raise AssertionError("ImageNet ViT-reference VR4 final/protocol identity mismatch")
    if final.get("paper_claim_authorized") is not primary_pass:
        raise AssertionError("ImageNet ViT-reference VR4 paper-claim decision changed")
    if final.get("claim_scope") != "cross_architecture_vit_reference_transfer":
        raise AssertionError("ImageNet ViT-reference VR4 final claim scope changed")
    if final.get("architecture_roles") != protocol["architecture_roles"]:
        raise AssertionError("ImageNet ViT-reference VR4 final architecture roles changed")
    if final.get("experiment_scope") != protocol["experiment_scope"]:
        raise AssertionError("ImageNet ViT-reference VR4 final experiment scope changed")
    if final.get("robust_vit_claim_authorized") is not False:
        raise AssertionError("ImageNet ViT-reference VR4 final permits a robust-ViT claim")
    if final.get("paper_claim_handoff_hash") != handoff_hash:
        raise AssertionError("ImageNet ViT-reference VR4 final/handoff identity mismatch")
    for key in (
        "confirmatory_attack_selection_performed",
        "confirmatory_seed_selection_performed",
        "confirmatory_per_example_selection_performed",
        "defense_retuning_performed",
    ):
        if final.get(key) is not False:
            raise AssertionError(f"ImageNet ViT-reference VR4 final reports forbidden action: {key}")
    for name, expected_hash in final.get("files", {}).items():
        path = directory / name
        if not path.is_file() or common.file_sha256(path) != expected_hash:
            raise AssertionError(f"ImageNet ViT-reference VR4 final file hash mismatch: {name}")
    return expected_methods, primary


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    output_root = args.output_root.expanduser().resolve()
    r0_dir = args.r0_dir.expanduser().resolve()
    calibration_dir = args.calibration_dir.expanduser().resolve()
    clean_dir = args.clean_dir.expanduser().resolve()
    vr3_root = args.vr3_root.expanduser().resolve()
    protocol, _parent, rows, inputs = verify_protocol_and_preparation(
        output_root, r0_dir, calibration_dir, clean_dir, vr3_root
    )
    clean = load_clean_images_for_audit(protocol, rows, inputs, args.num_workers)
    records = verify_search_cells(protocol, rows, inputs, clean, output_root)
    method_summary, primary = verify_analysis(protocol, output_root)
    print(
        f"PASS {output_root}: {len(records)} frozen attack cells, "
        "12,000 common-MC rows, candidates feasible",
        flush=True,
    )
    print("ImageNet ViT-reference VR4 confirmatory summary", flush=True)
    for row in method_summary:
        print(
            f"  {row['method']:>11}: {row['attack']} "
            f"Exact5={float(row['attacked_exact_set_k5_probability_percent']):.3f}% "
            f"sd={float(row['attacked_exact_set_k5_seed_sd_pp']):.3f} pp",
            flush=True,
        )
    print(
        f"  PRIMARY Full-Robust={float(primary['difference_pp']):+.3f} pp "
        f"[{float(primary['ci95_low_pp']):+.3f}, "
        f"{float(primary['ci95_high_pp']):+.3f}] -> {primary['status']}",
        flush=True,
    )
    print("PASS attacks were frozen on VR3; no VR4 selection or retuning occurred", flush=True)
    print(
        f"PAPER_CLAIM_AUTHORIZED={'YES' if primary['status'] == 'positive' else 'NO'}",
        flush=True,
    )
    print("ROBUST_VIT_CLAIM_AUTHORIZED=NO", flush=True)
    print(
        "ImageNet ViT-reference VR4 audit complete; no training, retuning, merge, deletion, or selection was performed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
