#!/usr/bin/env python3
"""Strict structural and scientific-protocol audit for CIFAR-100 R2."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms

import topk_cifar100_r2_common as protocol
import train_resnet20_dual_backbone_top2_cifar100_r2 as dual
import train_resnet20_teacher_top2_adversarial_cifar100_r2 as anchor_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CIFAR-100 model prep R2")
    parser.add_argument("--root", type=Path, default=protocol.DEFAULT_ROOT)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def require_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return protocol.read_json(path)


def finite_percent(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise AssertionError(f"{label} is not a finite percentage: {value}")
    return result


def read_epoch_rows(path: Path, expected: int) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    if len(rows) != expected:
        raise AssertionError(f"{path}: expected {expected} rows, got {len(rows)}")
    if len(set(epochs)) != len(epochs) or set(epochs) != set(range(1, expected + 1)):
        raise AssertionError(f"{path}: epochs are duplicated, missing, or non-contiguous")
    return {int(row["epoch"]): row for row in rows}


def assert_close(actual: Any, expected: Any, label: str, tol: float = 1e-8) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label}: {actual} != {expected}")


def audit_selected_checkpoint(metrics: dict[str, Any], stage: str) -> Path:
    if metrics.get("protocol_id") != protocol.PROTOCOL_ID:
        raise AssertionError(f"{stage}: wrong protocol id")
    if metrics.get("status") != "PASS" or metrics.get("test_used_for_selection") is not False:
        raise AssertionError(f"{stage}: invalid completion or test-selection marker")
    path = Path(str(metrics["selected_checkpoint"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{stage}: selected checkpoint missing: {path}")
    selected = metrics.get("selected_validation")
    if not isinstance(selected, dict):
        raise AssertionError(f"{stage}: no coherent selected-validation record")
    return path


def audit_model_weights(standard_path: Path, anchor_path: Path, dual_path: Path) -> None:
    device = torch.device("cpu")
    standard = anchor_code.make_model()
    anchor = anchor_code.make_model()
    anchor_code.load_model_weights(standard, standard_path, device, "R2 standard")
    anchor_code.load_model_weights(anchor, anchor_path, device, "R2 anchor")

    defense = dual.DualBackboneDefense()
    selected_state = dual.torch_load_compat(dual_path, device)
    defense.load_state_dict(selected_state)
    expected_robust = dual.CIFARResNet20()
    dual.load_body(expected_robust, anchor_path, device, "R2 anchor audit")
    for name, value in expected_robust.state_dict().items():
        observed = defense.robust_branch.state_dict()[name]
        if not torch.equal(value, observed):
            raise AssertionError(f"Dual frozen robust branch drifted at tensor {name}")
    if sum(parameter.numel() for parameter in standard.parameters()) <= 250_000:
        raise AssertionError("R2 standard architecture has unexpectedly few parameters")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = require_dict(root / "pipeline_manifest.json")
    if manifest.get("protocol_id") != protocol.PROTOCOL_ID:
        raise AssertionError("Pipeline manifest protocol mismatch")
    if manifest.get("incoming_pretrained_checkpoint") is not None:
        raise AssertionError("R2 must not consume a pretrained standard checkpoint")
    if manifest.get("downloads_allowed") is not False:
        raise AssertionError("R2 manifest does not prohibit downloads")

    preflight = require_dict(root / "preflight/preflight_report.json")
    if preflight.get("status") != "PASS" or preflight.get("incoming_checkpoint_used") is not False:
        raise AssertionError("Preflight did not certify the leakage-free offline protocol")
    data_root = protocol.resolve_torchvision_root(Path(manifest["data_dir"]))
    dataset = torchvision.datasets.CIFAR100(
        root=str(data_root), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    train_indices, validation_indices, split = protocol.load_frozen_split(
        root / "frozen_split.json", dataset.targets,
        int(manifest["seed"]), protocol.DEFAULT_VAL_SIZE,
    )
    if len(train_indices) != 45_000 or len(validation_indices) != 5_000:
        raise AssertionError("R2 frozen split is not 45k/5k")

    standard = require_dict(root / "standard/final_test_metrics.json")
    anchor = require_dict(root / "teacher_top2_at/final_test_metrics.json")
    defense = require_dict(root / "dual_top2/final_test_metrics.json")
    standard_path = audit_selected_checkpoint(standard, "standard")
    anchor_path = audit_selected_checkpoint(anchor, "anchor")
    dual_path = audit_selected_checkpoint(defense, "dual")
    for name, metrics in (("standard", standard), ("anchor", anchor), ("dual", defense)):
        if metrics.get("train_indices_sha256") != split["train_indices_sha256"]:
            raise AssertionError(f"{name}: training split hash drift")
        if metrics.get("validation_indices_sha256") != split["validation_indices_sha256"]:
            raise AssertionError(f"{name}: validation split hash drift")
        if metrics.get("official_full_test") is args.quick:
            raise AssertionError(f"{name}: official-test completeness marker is wrong")

    expected_standard_epochs = 2 if args.quick else 200
    expected_other_epochs = 2 if args.quick else 40
    standard_rows = read_epoch_rows(
        root / "standard/training_metrics.csv", expected_standard_epochs
    )
    anchor_rows = read_epoch_rows(
        root / "teacher_top2_at/training_metrics.csv", expected_other_epochs
    )
    dual_rows = read_epoch_rows(
        root / "dual_top2/training_metrics.csv", expected_other_epochs
    )

    standard_selected = standard["selected_validation"]
    standard_epoch = int(standard["selected_epoch"])
    standard_row = standard_rows[standard_epoch]
    assert_close(standard_selected["validation_top1"], standard_row["validation_top1"], "standard selected Top1")
    assert_close(standard_selected["validation_loss"], standard_row["validation_loss"], "standard selected loss")
    standard_val_top1 = finite_percent(standard_selected["validation_top1"], "standard val Top1")
    standard_test_top1 = finite_percent(standard["test_clean"]["top1"], "standard test Top1")
    if not args.quick and standard_val_top1 < 55.0:
        raise AssertionError("Standard validation Top-1 failed the frozen 55% gate")
    if int(standard["test_clean"]["examples"]) != (500 if args.quick else 10_000):
        raise AssertionError("Standard test example count is wrong")

    anchor_selected = anchor["selected_validation"]
    anchor_epoch = int(anchor["selected_epoch"])
    anchor_row = anchor_rows[anchor_epoch]
    if str(anchor_row["selection_eligible"]).lower() != "true":
        raise AssertionError("Anchor selected an ineligible warmup epoch")
    if not bool(anchor_selected.get("selection_eligible")):
        raise AssertionError("Anchor selected-validation record is ineligible")
    assert_close(anchor_selected["training_epsilon"], 2.0 / 255.0, "anchor selected epsilon", 1e-12)
    for field in (
        "validation_clean_top1", "validation_clean_set2",
        "validation_robust_top1", "validation_robust_set2",
        "validation_composite",
    ):
        assert_close(anchor_selected[field], anchor_row[field], f"anchor {field}")
    anchor_expected_composite = (
        0.20 * float(anchor_selected["validation_clean_top1"])
        + 0.20 * float(anchor_selected["validation_clean_set2"])
        + 0.30 * float(anchor_selected["validation_robust_top1"])
        + 0.30 * float(anchor_selected["validation_robust_set2"])
    )
    assert_close(anchor_selected["validation_composite"], anchor_expected_composite, "anchor composite")
    anchor_pgd_top1 = finite_percent(anchor_selected["validation_robust_top1"], "anchor val PGD Top1")
    anchor_pgd_set2 = finite_percent(anchor_selected["validation_robust_set2"], "anchor val PGD Set2")
    if not args.quick and (anchor_pgd_top1 <= 0.0 or anchor_pgd_set2 <= 0.0):
        raise AssertionError("Anchor failed the pre-registered non-degeneracy gate")

    dual_selected = defense["selected_validation"]
    dual_epoch = int(defense["selected_epoch"])
    if not bool(dual_selected.get("selection_eligible")):
        raise AssertionError("Dual selected-validation record is ineligible")
    if not bool(dual_selected.get("constraint_satisfied")):
        raise AssertionError("Dual selected a checkpoint below its clean Set2 floor")
    if float(dual_selected["validation_clean_set2"]) + 1e-10 < float(dual_selected["clean_set2_constraint"]):
        raise AssertionError("Dual selected clean Set2 violates the frozen floor")
    if dual_epoch > 0:
        dual_row = dual_rows[dual_epoch]
        if str(dual_row["selection_eligible"]).lower() != "true":
            raise AssertionError("Dual selected CSV row is ineligible")
        for field in (
            "validation_clean_top1", "validation_clean_set2",
            "validation_robust_top1", "validation_robust_set2",
            "validation_composite",
        ):
            row_field = "validation_final_set2" if field == "validation_clean_set2" else field
            if field == "validation_clean_top1":
                row_field = "validation_top1"
            assert_close(dual_selected[field], dual_row[row_field], f"dual {field}")
    dual_expected_composite = (
        0.20 * float(dual_selected["validation_clean_top1"])
        + 0.20 * float(dual_selected["validation_clean_set2"])
        + 0.30 * float(dual_selected["validation_robust_top1"])
        + 0.30 * float(dual_selected["validation_robust_set2"])
    )
    assert_close(dual_selected["validation_composite"], dual_expected_composite, "dual composite")

    standard_handoff = require_dict(root / "standard_to_downstream_handoff.json")
    anchor_handoff = require_dict(root / "teacher_to_dual_handoff.json")
    if standard_handoff["checkpoint_sha256"] != protocol.sha256_file(standard_path):
        raise AssertionError("Standard handoff checkpoint hash mismatch")
    if anchor_handoff["checkpoint_sha256"] != protocol.sha256_file(anchor_path):
        raise AssertionError("Anchor handoff checkpoint hash mismatch")
    audit_model_weights(standard_path, anchor_path, dual_path)

    dual_test_clean = defense["test_clean"]
    dual_test_robust = defense["test_robust"]
    report = {
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "PASS",
        "quick": args.quick,
        "split": {
            "train_indices_sha256": split["train_indices_sha256"],
            "validation_indices_sha256": split["validation_indices_sha256"],
        },
        "standard": {
            "selected_epoch": standard_epoch,
            "validation_top1": standard_val_top1,
            "test_top1": standard_test_top1,
        },
        "anchor": {
            "selected_epoch": anchor_epoch,
            "validation_pgd_top1": anchor_pgd_top1,
            "validation_pgd_exact_set2": anchor_pgd_set2,
            "test_clean_top1": finite_percent(anchor["test_clean"]["student_top1"], "anchor test clean Top1"),
            "test_clean_exact_set2": finite_percent(anchor["test_clean"]["set2_exact"], "anchor test clean Set2"),
            "test_pgd_top1": finite_percent(anchor["test_robust"]["robust_top1"], "anchor test PGD Top1"),
            "test_pgd_exact_set2": finite_percent(anchor["test_robust"]["robust_set2_exact"], "anchor test PGD Set2"),
        },
        "dual": {
            "selected_epoch": dual_epoch,
            "validation_clean_exact_set2": float(dual_selected["validation_clean_set2"]),
            "clean_constraint": float(dual_selected["clean_set2_constraint"]),
            "test_clean_top1": finite_percent(dual_test_clean["top1"], "dual test clean Top1"),
            "test_clean_exact_set2": finite_percent(dual_test_clean["final_set2_exact"], "dual test clean Set2"),
            "test_pgd_top1": finite_percent(dual_test_robust["robust_top1"], "dual test PGD Top1"),
            "test_pgd_exact_set2": finite_percent(dual_test_robust["robust_set2_exact"], "dual test PGD Set2"),
        },
        "next_gate": "calibration and adaptive black-box pilot only if this audit passes",
    }
    protocol.atomic_json(root / "audit_report.json", report)
    print(
        f"PASS CIFAR-100 R2: standard test Top1={standard_test_top1:.3f}%, "
        f"anchor PGD=({report['anchor']['test_pgd_top1']:.3f}% Top1, "
        f"{report['anchor']['test_pgd_exact_set2']:.3f}% ExactSet2), "
        f"Dual clean ExactSet2={report['dual']['test_clean_exact_set2']:.3f}%"
    )
    print("PASS the same immutable 45k/5k split is used by all three stages")
    print("PASS warmup checkpoints are excluded from anchor selection")
    print("PASS Dual selected checkpoint satisfies its frozen clean-fidelity floor")
    print("PASS Dual robust branch is bit-exact to the selected adversarial anchor")
    print("CIFAR-100 R2 audit complete; no merge, deletion, or attack selection was performed.")


if __name__ == "__main__":
    main()
