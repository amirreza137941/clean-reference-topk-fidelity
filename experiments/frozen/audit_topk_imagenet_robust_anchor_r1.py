#!/usr/bin/env python3
"""Independent, torch-free audit of ImageNet robust-anchor R1 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from topk_imagenet_robust_anchor_common_r1 import (
    EXPECTED_ARCHITECTURE_ID,
    EXPECTED_FAMILY,
    EXPECTED_SANITY_EXAMPLES,
    FROZEN_SPLIT_SHA256,
    PARENT_PROTOCOL_ID,
    PROTOCOL_ID,
    ROBUSTBENCH_EPSILON,
    ROBUSTBENCH_GDRIVE_ID,
    ROBUSTBENCH_MODEL_ID,
    STANDARD_CHECKPOINT_SHA256,
    atomic_json,
    read_csv,
    read_json,
    script_hashes,
    sha256_file,
)


DEFAULT_OUTPUT_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_robust_anchor_r1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ImageNet robust-anchor R1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit_sanity(output: Path, report: Mapping[str, Any], r0_dir: Path) -> dict[str, Any]:
    path = output / "sanity_predictions_salman2020do_r50.csv"
    rows = read_csv(path)
    require(len(rows) == EXPECTED_SANITY_EXAMPLES, "Robust sanity must contain 1,000 rows")
    require(len({row["relative_path"] for row in rows}) == len(rows), "Duplicate sanity paths")
    require({row["partition"] for row in rows} == {"preflight_sanity"}, "Wrong sanity partition")
    top1 = sum(int(row["top1_correct"]) for row in rows)
    top5 = sum(int(row["top5_correct"]) for row in rows)
    sanity = report["sanity"]
    require(int(sanity["top1_correct"]) == top1, "Sanity Top-1 aggregate mismatch")
    require(int(sanity["top5_correct"]) == top5, "Sanity Top-5 aggregate mismatch")
    require(close(float(sanity["top1_percent"]), 100.0 * top1 / len(rows)), "Sanity Top-1 percent mismatch")
    require(close(float(sanity["top5_percent"]), 100.0 * top5 / len(rows)), "Sanity Top-5 percent mismatch")
    require(sanity["gate"] == "PASS", "Robust clean sanity gate failed")
    require(sha256_file(path) == sanity["prediction_csv_sha256"], "Sanity CSV SHA mismatch")

    parent_rows = read_csv(r0_dir / "sanity_predictions_resnet50_local.csv")
    require(len(parent_rows) == len(rows), "R0/R1 sanity row counts differ")
    parent_map = {row["relative_path"]: row for row in parent_rows}
    robust_map = {row["relative_path"]: row for row in rows}
    require(parent_map.keys() == robust_map.keys(), "R0/R1 sanity samples differ")
    for relative_path in parent_map:
        require(
            parent_map[relative_path]["target"] == robust_map[relative_path]["target"],
            f"R0/R1 target mismatch at {relative_path}",
        )
    return {
        "examples": len(rows),
        "top1_percent": 100.0 * top1 / len(rows),
        "top5_percent": 100.0 * top5 / len(rows),
    }


def audit_pgd(output: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    pgd = report["pgd_identity"]
    path = output / "pgd_identity_sanity.csv"
    rows = read_csv(path)
    require(len(rows) == int(pgd["examples"]), "PGD row count mismatch")
    require(len(rows) > 0, "PGD identity table is empty")
    require(len({row["relative_path"] for row in rows}) == len(rows), "Duplicate PGD paths")
    require({row["partition"] for row in rows} == {"preflight_sanity"}, "Wrong PGD partition")
    require(
        {row["selection"] for row in rows} == {"outcome_blind_hash_rank"},
        "PGD sample selection was not frozen/outcome-blind",
    )
    require(sha256_file(path) == pgd["rows_csv_sha256"], "PGD CSV SHA mismatch")
    require(close(float(pgd["epsilon_linf_pixel"]), ROBUSTBENCH_EPSILON), "PGD epsilon mismatch")
    require(float(pgd["epsilon_linf_255"]) == 4.0, "PGD epsilon/255 mismatch")
    require(pgd["paper_claim_authorized"] is False, "Diagnostic PGD cannot authorize a paper claim")
    require(pgd["gate"] == "PASS", "PGD identity gate failed")

    aggregate: dict[str, dict[str, float]] = {}
    for method in ("standard", "robust"):
        clean = sum(int(row[f"{method}_clean_correct"]) for row in rows)
        attacked = sum(int(row[f"{method}_pgd_correct"]) for row in rows)
        maximum = max(float(row[f"{method}_linf"]) for row in rows)
        expected = pgd["methods"][method]
        require(
            close(float(expected["clean_top1_percent"]), 100.0 * clean / len(rows)),
            f"{method} PGD-table clean aggregate mismatch",
        )
        require(
            close(float(expected["pgd_top1_percent"]), 100.0 * attacked / len(rows)),
            f"{method} PGD-table attacked aggregate mismatch",
        )
        require(
            close(float(expected["maximum_linf"]), maximum, tolerance=1e-7),
            f"{method} maximum Linf mismatch",
        )
        require(maximum <= ROBUSTBENCH_EPSILON + 1e-6, f"{method} PGD is infeasible")
        aggregate[method] = {
            "clean_top1_percent": 100.0 * clean / len(rows),
            "pgd_top1_percent": 100.0 * attacked / len(rows),
            "maximum_linf": maximum,
        }
    advantage = aggregate["robust"]["pgd_top1_percent"] - aggregate["standard"][
        "pgd_top1_percent"
    ]
    require(
        close(float(pgd["robust_minus_standard_pgd_top1_pp"]), advantage),
        "PGD robust-standard advantage mismatch",
    )
    require(
        aggregate["robust"]["pgd_top1_percent"]
        >= float(pgd["minimum_robust_pgd_top1_percent"]),
        "Robust PGD accuracy floor failed",
    )
    require(
        advantage >= float(pgd["minimum_robust_advantage_pp"]),
        "Robust-standard PGD advantage floor failed",
    )
    return {"examples": len(rows), "methods": aggregate, "advantage_pp": advantage}


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    run = read_json(output / "run_manifest.json")
    report = read_json(output / "robust_anchor_report.json")
    roles = read_json(output / "checkpoint_role_assignment_frozen_r1.json")
    handoff = read_json(output / "calibration_handoff_r1.json")

    require(run.get("protocol_id") == PROTOCOL_ID, "Run protocol mismatch")
    require(run.get("parent_protocol_id") == PARENT_PROTOCOL_ID, "Parent protocol mismatch")
    require(run.get("frozen_split_sha256") == FROZEN_SPLIT_SHA256, "Run split hash mismatch")
    require(run.get("dataset_download_allowed") is False, "R1 must not download ImageNet")
    require(run.get("training_allowed") is False, "R1 must not train")
    require(run.get("defense_tuning_allowed") is False, "R1 must not tune the defense")
    require(run.get("paper_attack_claim_allowed") is False, "R1 PGD is diagnostic only")
    require(run.get("outcomes_read_from_partitions") == ["preflight_sanity"], "R1 read an unauthorized split")
    require(
        set(run.get("outcome_blind_partitions", []))
        == {"calibration_dev", "attack_pilot", "attack_confirmatory", "clean_confirmation"},
        "Outcome-blind partition declaration mismatch",
    )
    source = run["source"]
    require(source["model_id"] == ROBUSTBENCH_MODEL_ID, "RobustBench model id mismatch")
    require(source["google_drive_id"] == ROBUSTBENCH_GDRIVE_ID, "Google Drive id mismatch")
    require(float(source["epsilon_linf_pixel"]) == ROBUSTBENCH_EPSILON, "Source epsilon mismatch")

    paths = {key: Path(value) for key, value in run["paths"].items()}
    current_hashes = script_hashes(paths["project_dir"])
    require(current_hashes == run["script_sha256"], "R1/R0 source files changed after execution")
    require(sha256_file(paths["standard_checkpoint"]) == STANDARD_CHECKPOINT_SHA256, "Standard checkpoint changed")
    require(
        sha256_file(paths["raw_checkpoint"]) == report["raw_checkpoint"]["sha256"],
        "Raw robust checkpoint changed",
    )
    require(
        sha256_file(paths["export_checkpoint"])
        == report["exported_checkpoint"]["sha256"],
        "Exported robust checkpoint changed",
    )
    require(
        report["raw_checkpoint"]["sha256"] != STANDARD_CHECKPOINT_SHA256,
        "Raw robust and standard hashes are identical",
    )
    require(
        report["exported_checkpoint"]["sha256"] != STANDARD_CHECKPOINT_SHA256,
        "Exported robust and standard hashes are identical",
    )
    load = report["robust_checkpoint_load"]
    require(load["architecture"]["architecture_id"] == EXPECTED_ARCHITECTURE_ID, "Robust architecture mismatch")
    require(load["architecture"]["family"] == EXPECTED_FAMILY, "Robust family mismatch")
    require(float(load["match"]["parameter_numel_coverage"]) == 1.0, "Robust parameter coverage is not 100%")
    require(float(load["match"]["numel_coverage"]) == 1.0, "Robust state coverage is not 100%")
    require(not load["match"]["missing_keys"], "Robust checkpoint has missing keys")
    require(not load["match"]["shape_mismatches"], "Robust checkpoint has shape mismatches")
    require(
        report["exported_checkpoint"]["state_dict_fingerprint_sha256"]
        == load["state_dict_fingerprint_sha256"],
        "Raw/export tensor fingerprints differ",
    )

    r0_dir = paths["r0_dir"]
    parent_manifest = read_json(r0_dir / "run_manifest.json")
    split_manifest = read_json(r0_dir / "frozen_imagenet_val_split_manifest.json")
    require(parent_manifest["protocol_id"] == PARENT_PROTOCOL_ID, "Live R0 protocol changed")
    require(parent_manifest["frozen_split_sha256"] == FROZEN_SPLIT_SHA256, "Live R0 split changed")
    require(split_manifest["split_sha256"] == FROZEN_SPLIT_SHA256, "Live split manifest changed")

    sanity = audit_sanity(output, report, r0_dir)
    pgd = audit_pgd(output, report)

    require(report.get("status") == "PASS", "Robust anchor report is not PASS")
    require(report.get("defense_calibration_ready") is True, "Calibration is not ready")
    require(not report.get("blockers"), "Robust anchor report has blockers")
    require(roles.get("status") == "FROZEN", "Checkpoint roles are not frozen")
    require(roles.get("frozen_split_sha256") == FROZEN_SPLIT_SHA256, "Role split mismatch")
    require(roles["standard_reference"]["sha256"] == STANDARD_CHECKPOINT_SHA256, "Role standard SHA mismatch")
    require(
        roles["robust_anchor"]["sha256"] == report["exported_checkpoint"]["sha256"],
        "Role robust SHA mismatch",
    )
    require(roles.get("same_architecture_family") is True, "Roles are not same-family")
    require(roles.get("distinct_checkpoint_tensors") is True, "Roles are not tensor-distinct")
    require(sha256_file(output / "checkpoint_role_assignment_frozen_r1.json") == report["role_assignment_sha256"], "Role JSON SHA mismatch")
    require(handoff.get("status") == "READY", "Calibration handoff is not READY")
    require(handoff.get("calibration_partition") == "calibration_dev", "Wrong calibration partition")
    require(int(handoff.get("calibration_examples")) == 2000, "Wrong calibration size")
    require(handoff.get("do_not_transfer_cifar_hyperparameters") is True, "CIFAR transfer guard missing")

    audit = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "output_dir": str(output),
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "model_id": ROBUSTBENCH_MODEL_ID,
        "raw_checkpoint_sha256": report["raw_checkpoint"]["sha256"],
        "exported_checkpoint_sha256": report["exported_checkpoint"]["sha256"],
        "architecture_id": EXPECTED_ARCHITECTURE_ID,
        "sanity": sanity,
        "pgd_identity": pgd,
        "role_assignment": "FROZEN",
        "defense_calibration_ready": True,
        "paper_attack_claim_authorized": False,
    }
    atomic_json(output / "audit_report.json", audit)
    print(
        f"PASS {output}: official {ROBUSTBENCH_MODEL_ID}, exact {EXPECTED_ARCHITECTURE_ID}, "
        "roles frozen"
    )
    print(
        f"  clean sanity: Top1={sanity['top1_percent']:.3f}% "
        f"Top5={sanity['top5_percent']:.3f}%"
    )
    print(
        f"  diagnostic PGD Top1: standard={pgd['methods']['standard']['pgd_top1_percent']:.3f}% "
        f"robust={pgd['methods']['robust']['pgd_top1_percent']:.3f}% "
        f"delta={pgd['advantage_pp']:+.3f} pp"
    )
    print("ROLE_ASSIGNMENT=FROZEN")
    print("DEFENSE_CALIBRATION_READY=YES")
    print("PAPER_ATTACK_CLAIM_AUTHORIZED=NO (identity sanity only)")
    print("R1 audit complete; ImageNet was not downloaded or modified.")


if __name__ == "__main__":
    main()
