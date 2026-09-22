#!/usr/bin/env python3
"""Independent structural and numerical audit for ImageNet Top-k R0 preflight."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from topk_imagenet_preflight_common import (
    EXPECTED_TRAIN_EXAMPLES,
    EXPECTED_VAL_EXAMPLES,
    NUM_CLASSES,
    PROTOCOL_ID,
    SPLIT_SPEC,
    atomic_json,
    partition_rows,
    read_csv,
    read_json,
    sha256_file,
    validate_frozen_split,
)


DEFAULT_OUTPUT = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_preflight_r0"
)
PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ImageNet Top-k R0 preflight")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def canonical_split_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "partition": row["partition"],
            "target": int(row["target"]),
            "class_name": row["class_name"],
            "within_class_slot": int(row["within_class_slot"]),
            "relative_path": row["relative_path"],
        }
        for row in raw_rows
    ]


def audit_scripts(manifest: dict) -> None:
    expected = manifest.get("script_sha256")
    if not isinstance(expected, dict) or not expected:
        raise AssertionError("run_manifest has no script hashes")
    for name, digest in expected.items():
        path = PROJECT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        assert_equal(sha256_file(path), digest, f"script SHA256 {name}")


def audit_split(output_dir: Path, report: dict, manifest: dict) -> tuple[list[dict[str, object]], dict]:
    split_path = output_dir / "frozen_imagenet_val_split.csv"
    split_manifest_path = output_dir / "frozen_imagenet_val_split_manifest.json"
    rows = canonical_split_rows(read_csv(split_path))
    summary = validate_frozen_split(rows)
    frozen = read_json(split_manifest_path)
    assert_equal(frozen.get("protocol_id"), PROTOCOL_ID, "split protocol")
    assert_equal(frozen.get("created_before_model_evaluation"), True, "outcome-blind creation")
    assert_equal(frozen.get("outcome_blind"), True, "outcome-blind flag")
    assert_equal(frozen.get("split_spec_per_class"), dict(SPLIT_SPEC), "split spec")
    assert_equal(frozen.get("split_sha256"), summary["split_sha256"], "split SHA256")
    assert_equal(
        manifest.get("frozen_split_sha256"), summary["split_sha256"], "manifest split SHA256"
    )
    assert_equal(
        report["dataset"]["split"]["split_sha256"],
        summary["split_sha256"],
        "report split SHA256",
    )
    return rows, summary


def audit_predictions(
    output_dir: Path,
    checkpoint: dict,
    sanity_rows: list[dict[str, object]],
) -> dict[str, object]:
    checkpoint_id = str(checkpoint["checkpoint_id"])
    if checkpoint["status"] != "PASS" and "sanity" not in checkpoint:
        expected = output_dir / f"sanity_predictions_{checkpoint_id}.csv"
        if expected.exists():
            raise AssertionError(
                f"Blocked checkpoint unexpectedly has a prediction file: {expected}"
            )
        return {"checkpoint_id": checkpoint_id, "status": "BLOCKED_AS_REPORTED"}

    architecture = checkpoint.get("architecture", {})
    match = checkpoint.get("state_dict_match", {})
    if architecture.get("family") not in {"resnet50", "vit_b16"}:
        raise AssertionError(f"Unexpected architecture family for {checkpoint_id}")
    if not bool(match.get("load_ready")):
        raise AssertionError(f"{checkpoint_id} was marked PASS without load_ready")
    if float(match.get("parameter_numel_coverage", 0.0)) < 0.995:
        raise AssertionError(f"Low parameter coverage for {checkpoint_id}")
    if float(match.get("numel_coverage", 0.0)) < 0.995:
        raise AssertionError(f"Low state coverage for {checkpoint_id}")
    if not bool(match.get("classifier_complete")):
        raise AssertionError(f"Incomplete classifier for {checkpoint_id}")

    sanity = checkpoint["sanity"]
    path = Path(sanity["prediction_csv"]).expanduser().resolve()
    expected_path = (output_dir / f"sanity_predictions_{checkpoint_id}.csv").resolve()
    assert_equal(path, expected_path, f"prediction path {checkpoint_id}")
    assert_equal(sha256_file(path), sanity["prediction_csv_sha256"], f"prediction hash {checkpoint_id}")
    rows = read_csv(path)
    assert_equal(len(rows), NUM_CLASSES, f"prediction count {checkpoint_id}")
    if set(row["partition"] for row in rows) != {"preflight_sanity"}:
        raise AssertionError(f"{checkpoint_id} evaluated a non-sanity partition")
    row_paths = [row["relative_path"] for row in rows]
    expected_paths = [str(row["relative_path"]) for row in sanity_rows]
    assert_equal(set(row_paths), set(expected_paths), f"sanity path set {checkpoint_id}")
    if len(row_paths) != len(set(row_paths)):
        raise AssertionError(f"Duplicate prediction paths for {checkpoint_id}")
    target_by_path = {
        str(row["relative_path"]): int(row["target"]) for row in sanity_rows
    }
    for row in rows:
        assert_equal(
            int(row["target"]),
            target_by_path[row["relative_path"]],
            f"target {checkpoint_id}/{row['relative_path']}",
        )
        for field in ("logit_min", "logit_max", "logit_mean", "logit_std"):
            if not math.isfinite(float(row[field])):
                raise AssertionError(f"Non-finite {field} for {checkpoint_id}")
    top1_correct = sum(int(row["top1_correct"]) for row in rows)
    top5_correct = sum(int(row["top5_correct"]) for row in rows)
    assert_equal(top1_correct, int(sanity["top1_correct"]), f"Top-1 count {checkpoint_id}")
    assert_equal(top5_correct, int(sanity["top5_correct"]), f"Top-5 count {checkpoint_id}")
    assert_close(100.0 * top1_correct / len(rows), sanity["top1_percent"], f"Top-1 {checkpoint_id}")
    assert_close(100.0 * top5_correct / len(rows), sanity["top5_percent"], f"Top-5 {checkpoint_id}")
    if checkpoint["status"] == "PASS":
        assert_equal(sanity["sanity_gate"], "PASS", f"sanity gate {checkpoint_id}")
    elif sanity["sanity_gate"] not in {"PASS", "FAIL"}:
        raise AssertionError(f"Unknown sanity gate for blocked {checkpoint_id}")

    memory = checkpoint.get("memory_latency_224", {})
    if memory.get("status") == "PASS":
        memory_rows = memory.get("rows", [])
        if not memory_rows:
            raise AssertionError(f"Empty memory sweep for {checkpoint_id}")
        if not any(row.get("forward_status") == "PASS" for row in memory_rows):
            raise AssertionError(f"No passing forward batch for {checkpoint_id}")
        if not any(row.get("backward_status") == "PASS" for row in memory_rows):
            raise AssertionError(f"No passing backward batch for {checkpoint_id}")
    elif memory.get("status") not in {"SKIPPED_BY_FLAG", "SKIPPED_CPU", "ERROR"}:
        raise AssertionError(f"Unexpected memory status for {checkpoint_id}: {memory}")

    checkpoint_path = Path(checkpoint["path"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    assert_equal(sha256_file(checkpoint_path), checkpoint["sha256"], f"checkpoint hash {checkpoint_id}")
    return {
        "checkpoint_id": checkpoint_id,
        "status": checkpoint["status"],
        "architecture": architecture["architecture_id"],
        "top1_percent": sanity["top1_percent"],
        "top5_percent": sanity["top5_percent"],
    }


def audit_role_logic(report: dict, output_dir: Path) -> tuple[str, list[str]]:
    assignment_path = Path(report["role_assignment_path"]).expanduser().resolve()
    assert_equal(
        assignment_path,
        (output_dir / "checkpoint_role_assignment.json").resolve(),
        "role-assignment path",
    )
    assignment = read_json(assignment_path)
    checkpoints = {item["checkpoint_id"]: item for item in report["checkpoints"]}
    blockers: list[str] = []
    if assignment.get("protocol_id") != PROTOCOL_ID:
        blockers.append("role assignment protocol mismatch")
    if assignment.get("status") != "FROZEN":
        blockers.append("checkpoint roles are not explicitly FROZEN")
    standard_id = assignment.get("standard_reference_checkpoint_id")
    anchor_id = assignment.get("robust_anchor_checkpoint_id")
    if standard_id not in checkpoints:
        blockers.append("standard reference checkpoint id is missing or unknown")
    if anchor_id not in checkpoints:
        blockers.append("robust anchor checkpoint id is missing or unknown")
    if standard_id == anchor_id and standard_id is not None:
        blockers.append("standard and anchor checkpoints are not distinct")
    if standard_id in checkpoints and anchor_id in checkpoints:
        left = checkpoints[standard_id]
        right = checkpoints[anchor_id]
        if left["status"] != "PASS" or right["status"] != "PASS":
            blockers.append("one assigned checkpoint failed preflight")
        left_family = left["architecture"]["family"]
        right_family = right["architecture"]["family"]
        if left_family != right_family:
            blockers.append(
                f"architecture-family mismatch: standard={left_family}, anchor={right_family}"
            )
        if assignment.get("standard_reference_checkpoint_sha256") != left.get("sha256"):
            blockers.append("standard reference checkpoint SHA256 is absent or stale")
        if assignment.get("robust_anchor_checkpoint_sha256") != right.get("sha256"):
            blockers.append("robust anchor checkpoint SHA256 is absent or stale")
    provenance = assignment.get("robust_anchor_training_provenance")
    if not isinstance(provenance, str) or len(provenance.strip()) < 20:
        blockers.append("robust anchor provenance is absent or too vague")
    status = "PASS" if not blockers else "BLOCKED"
    assert_equal(report["role_assignment_status"], status, "reported role status")
    assert_equal(bool(report["calibration_ready"]), not blockers, "calibration readiness")
    if report.get("blockers") != [
        item for item in report.get("blockers", [])
    ]:
        raise AssertionError("Malformed blockers list")
    return status, blockers


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    report = read_json(output_dir / "preflight_report.json")
    manifest = read_json(output_dir / "run_manifest.json")
    assert_equal(report.get("protocol_id"), PROTOCOL_ID, "report protocol")
    assert_equal(manifest.get("protocol_id"), PROTOCOL_ID, "manifest protocol")
    assert_equal(report.get("preflight_complete"), True, "preflight-complete flag")
    assert_equal(report.get("downloads_attempted"), False, "download-attempt flag")
    assert_equal(manifest.get("downloads_allowed"), False, "download policy")
    assert_equal(manifest.get("training_allowed"), False, "training policy")
    assert_equal(manifest.get("tuning_allowed"), False, "tuning policy")
    assert_equal(manifest.get("attack_allowed"), False, "attack policy")
    assert_equal(report.get("outcomes_read_from_partitions"), ["preflight_sanity"], "read partitions")
    assert_equal(
        sorted(report.get("outcome_blind_partitions", [])),
        sorted(name for name in SPLIT_SPEC if name != "preflight_sanity"),
        "outcome-blind partitions",
    )
    audit_scripts(manifest)

    dataset = report["dataset"]
    assert_equal(dataset["status"], "PASS", "dataset status")
    assert_equal(dataset["train_classes"], NUM_CLASSES, "train classes")
    assert_equal(dataset["val_classes"], NUM_CLASSES, "val classes")
    assert_equal(dataset["train_examples"], EXPECTED_TRAIN_EXAMPLES, "train examples")
    assert_equal(dataset["val_examples"], EXPECTED_VAL_EXAMPLES, "val examples")
    assert_equal(dataset["val_per_class_values"], [50], "val class balance")
    if int(dataset["decode_audit"]["decoded"]) != 100:
        raise AssertionError("Exactly 100 validation images must be decode-audited")

    frozen_rows, split_summary = audit_split(output_dir, report, manifest)
    sanity_rows = partition_rows(frozen_rows, "preflight_sanity")
    checkpoint_audits = [
        audit_predictions(output_dir, item, sanity_rows)
        for item in report["checkpoints"]
    ]
    if len(checkpoint_audits) != 2:
        raise AssertionError("Expected exactly two local checkpoint audits")
    checkpoint_ids = [item["checkpoint_id"] for item in checkpoint_audits]
    assert_equal(checkpoint_ids, ["resnet50_local", "vit_b16_local"], "checkpoint order")

    role_status, role_blockers = audit_role_logic(report, output_dir)
    audit_payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "structural_preflight": "PASS",
        "split": split_summary,
        "checkpoints": checkpoint_audits,
        "role_assignment_status": role_status,
        "calibration_ready": report["calibration_ready"],
        "role_blockers_recomputed": role_blockers,
        "scientific_interpretation": (
            "Proceed to defense calibration only after a distinct same-family robust anchor "
            "with verified provenance is frozen."
        ),
    }
    atomic_json(output_dir / "audit_report.json", audit_payload)

    print(
        f"PASS {output_dir}: train={dataset['train_examples']:,}, "
        f"val={dataset['val_examples']:,}, frozen rows={len(frozen_rows):,}"
    )
    for item in checkpoint_audits:
        if item["status"] == "PASS":
            print(
                f"  {item['checkpoint_id']}: {item['architecture']}, "
                f"Top1={item['top1_percent']:.3f}% Top5={item['top5_percent']:.3f}%"
            )
        else:
            print(f"  {item['checkpoint_id']}: BLOCKED as reported")
    print(f"ROLE_ASSIGNMENT={role_status}")
    print(f"DEFENSE_CALIBRATION_READY={'YES' if report['calibration_ready'] else 'NO'}")
    if role_blockers:
        print("Expected role blockers:")
        for blocker in role_blockers:
            print(f"  - {blocker}")
    print("ImageNet R0 audit complete; no download, training, tuning, or attack was performed.")


if __name__ == "__main__":
    main()
