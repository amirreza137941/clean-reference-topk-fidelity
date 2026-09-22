#!/usr/bin/env python3
"""Run the frozen, leakage-free CIFAR-100 R2 preparation pipeline."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import topk_cifar100_r2_common as protocol


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight, train standard/anchor/Dual, then audit CIFAR-100 R2"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "preflight", "standard", "teacher", "dual", "audit"],
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=protocol.DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=protocol.DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=protocol.DEFAULT_SEED)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


@contextmanager
def exclusive_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".pipeline.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another R2 process holds {lock_path}") from exc
        yield


def selected_checkpoint(stage_dir: Path) -> Path | None:
    metrics_path = stage_dir / "final_test_metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = protocol.read_json(metrics_path)
    selected = Path(str(metrics["selected_checkpoint"])).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(
            f"Completion marker exists but selected checkpoint is absent: {selected}"
        )
    return selected


def frozen_manifest(args: argparse.Namespace, root: Path) -> dict:
    names = [
        "topk_cifar100_r2_common.py",
        "preflight_topk_cifar100_model_prep_r2.py",
        "train_resnet20_standard_cifar100_r2.py",
        "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
        "train_resnet20_dual_backbone_top2_cifar100_r2.py",
        "audit_topk_cifar100_model_prep_r2.py",
        "run_topk_cifar100_model_prep_r2.py",
    ]
    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "purpose": "SATML external-validity model preparation; attack evaluation follows only after this gate",
        "offline": True,
        "downloads_allowed": False,
        "incoming_pretrained_checkpoint": None,
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "num_classes": 100,
        "primary_k": 2,
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "output_root": str(root),
        "seed": args.seed,
        "quick": args.quick,
        "device": args.device,
        "amp_enabled": args.device.startswith("cuda") and not args.no_amp,
        "split": {
            "training_examples": 45_000,
            "validation_examples": 5_000,
            "rule": "fixed class-stratified 450/50 per class",
            "shared_across_all_three_training_stages": True,
        },
        "standard_training": {
            "random_initialization": True,
            "epochs": 200,
            "optimizer": "SGD",
            "initial_lr": 0.1,
            "schedule": "cosine",
            "selection": "maximum validation Top-1, tie broken by lower loss",
        },
        "training_threat": {
            "norm": "Linf",
            "epsilon_px": 2.0,
            "pgd_steps": 7,
            "pgd_alpha_px": 0.5,
        },
        "anchor_selection": {
            "metric": "0.20 clean Top-1 + 0.20 clean Exact-Set2 + 0.30 PGD Top-1 + 0.30 PGD Exact-Set2",
            "pre_full_radius_epochs_eligible": False,
        },
        "dual_selection": {
            "metric": "maximum fixed composite among clean-feasible candidates, always evaluated by full-radius PGD",
            "clean_exact_set_floor": "initial validation Exact-Set2 minus 2 pp",
            "positive_class_weight": 49.0,
        },
        "official_test_policy": "one evaluation per stage, after frozen validation selection",
        "script_sha256": {
            name: protocol.sha256_file(PROJECT_DIR / name)
            for name in names
            if (PROJECT_DIR / name).is_file()
        },
    }


def ensure_manifest(args: argparse.Namespace, root: Path) -> None:
    path = root / "pipeline_manifest.json"
    proposed = frozen_manifest(args, root)
    if path.is_file():
        if protocol.read_json(path) != proposed:
            raise RuntimeError(
                "Existing R2 manifest differs from this code/command. Resume with "
                "the original bundle and arguments, or choose a new output root."
            )
        print(f"PASS frozen R2 manifest unchanged: {path}")
    else:
        protocol.atomic_json(path, proposed)
        print(f"Frozen R2 manifest created: {path}")


def ensure_handoff(
    path: Path, stage: str, checkpoint: Path, split_file: Path
) -> None:
    split = protocol.read_json(split_file)
    proposed = {
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": stage,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": protocol.sha256_file(checkpoint),
        "split_file": str(split_file),
        "train_indices_sha256": split["train_indices_sha256"],
        "validation_indices_sha256": split["validation_indices_sha256"],
    }
    if path.is_file():
        if protocol.read_json(path) != proposed:
            raise RuntimeError(f"Frozen handoff changed: {path}")
    else:
        protocol.atomic_json(path, proposed)
    print(f"PASS frozen {stage} handoff: {proposed['checkpoint_sha256']}")


def run_preflight(args: argparse.Namespace, root: Path, split_file: Path) -> None:
    run([
        sys.executable, "-u",
        str(PROJECT_DIR / "preflight_topk_cifar100_model_prep_r2.py"),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(root / "preflight"),
        "--split-file", str(split_file),
        "--device", args.device,
        "--split-seed", str(args.seed),
    ])


def run_standard(args: argparse.Namespace, root: Path, split_file: Path) -> Path:
    output = root / "standard"
    selected = selected_checkpoint(output)
    if selected is not None:
        print(f"PASS standard stage already complete: {selected}")
        return selected
    command = [
        sys.executable, "-u",
        str(PROJECT_DIR / "train_resnet20_standard_cifar100_r2.py"),
        "--data-dir", str(args.data_dir),
        "--split-file", str(split_file),
        "--output-dir", str(output),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.device.startswith("cuda") and not args.no_amp:
        command.append("--amp")
    if args.quick:
        command.append("--quick")
    if (output / "last_checkpoint.pt").is_file():
        command.append("--resume")
    run(command)
    selected = selected_checkpoint(output)
    if selected is None:
        raise RuntimeError("Standard stage ended without a completion marker")
    return selected


def run_teacher(
    args: argparse.Namespace, root: Path, split_file: Path, standard: Path
) -> Path:
    output = root / "teacher_top2_at"
    selected = selected_checkpoint(output)
    if selected is not None:
        print(f"PASS anchor stage already complete: {selected}")
        return selected
    command = [
        sys.executable, "-u",
        str(PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py"),
        "--data-dir", str(args.data_dir),
        "--split-file", str(split_file),
        "--teacher-weights", str(standard),
        "--student-init-weights", str(standard),
        "--output-dir", str(output),
        "--device", args.device,
        "--seed", str(args.seed),
        "--epsilon", str(2.0 / 255.0),
        "--pgd-alpha", str(0.5 / 255.0),
        "--robust-eval-alpha", str(0.5 / 255.0),
    ]
    if args.device.startswith("cuda") and not args.no_amp:
        command.append("--amp")
    if args.quick:
        command.append("--quick")
    if (output / "last_checkpoint.pt").is_file():
        command.append("--resume")
    run(command)
    selected = selected_checkpoint(output)
    if selected is None:
        raise RuntimeError("Anchor stage ended without a completion marker")
    return selected


def run_dual(
    args: argparse.Namespace, root: Path, split_file: Path,
    standard: Path, anchor: Path,
) -> Path:
    output = root / "dual_top2"
    selected = selected_checkpoint(output)
    if selected is not None:
        print(f"PASS Dual stage already complete: {selected}")
        return selected
    command = [
        sys.executable, "-u",
        str(PROJECT_DIR / "train_resnet20_dual_backbone_top2_cifar100_r2.py"),
        "--data-dir", str(args.data_dir),
        "--split-file", str(split_file),
        "--teacher-weights", str(standard),
        "--robust-anchor-weights", str(anchor),
        "--output-dir", str(output),
        "--device", args.device,
        "--seed", str(args.seed),
        "--epsilon", str(2.0 / 255.0),
        "--pgd-alpha", str(0.5 / 255.0),
        "--robust-eval-alpha", str(0.5 / 255.0),
    ]
    if args.device.startswith("cuda") and not args.no_amp:
        command.append("--amp")
    if args.quick:
        command.append("--quick")
    if (output / "last_checkpoint.pt").is_file():
        command.append("--resume")
    run(command)
    selected = selected_checkpoint(output)
    if selected is None:
        raise RuntimeError("Dual stage ended without a completion marker")
    return selected


def run_audit(args: argparse.Namespace, root: Path) -> None:
    command = [
        sys.executable, "-u",
        str(PROJECT_DIR / "audit_topk_cifar100_model_prep_r2.py"),
        "--root", str(root),
    ]
    if args.quick:
        command.append("--quick")
    run(command)


def require_selected(root: Path, stage: str) -> Path:
    selected = selected_checkpoint(root / stage)
    if selected is None:
        raise FileNotFoundError(
            f"Required R2 stage is incomplete: {root / stage}. Run --stage all "
            "or the prerequisite stage first."
        )
    return selected


def validate_standard_gate(root: Path, quick: bool) -> None:
    metrics = protocol.read_json(root / "standard/final_test_metrics.json")
    selected = metrics.get("selected_validation")
    if not isinstance(selected, dict):
        raise RuntimeError("Standard stage lacks a coherent selected-validation record")
    if not quick and float(selected["validation_top1"]) < 55.0:
        raise RuntimeError(
            "R2 standard-reference gate failed: selected validation Top-1 is "
            f"{float(selected['validation_top1']):.3f}% (<55%). Stop rather than "
            "building the defense on a degenerate reference."
        )
    print(
        "PASS standard validation gate: "
        f"Top1={float(selected['validation_top1']):.3f}%"
    )


def validate_anchor_gate(root: Path, quick: bool) -> None:
    metrics = protocol.read_json(root / "teacher_top2_at/final_test_metrics.json")
    selected = metrics.get("selected_validation")
    if not isinstance(selected, dict) or not bool(selected.get("selection_eligible")):
        raise RuntimeError("Adversarial anchor was not selected from an eligible epoch")
    if abs(float(selected["training_epsilon"]) - 2.0 / 255.0) > 1e-12:
        raise RuntimeError("Adversarial anchor was selected below the frozen 2/255 radius")
    if not quick and (
        float(selected["validation_robust_top1"]) <= 0.0
        or float(selected["validation_robust_set2"]) <= 0.0
    ):
        raise RuntimeError(
            "R2 anchor non-degeneracy gate failed: full-radius validation PGD is "
            "zero. Stop before Dual; do not repeat the R1 failure."
        )
    print(
        "PASS full-radius anchor gate: "
        f"PGD Top1={float(selected['validation_robust_top1']):.3f}% "
        f"ExactSet2={float(selected['validation_robust_set2']):.3f}%"
    )


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Local CIFAR-100 directory not found: {args.data_dir}")
    root = args.output_root.expanduser().resolve()
    if args.quick and root == protocol.DEFAULT_ROOT:
        root = root.with_name(root.name + "_smoke")
    split_file = root / "frozen_split.json"

    with exclusive_lock(root):
        ensure_manifest(args, root)
        if args.stage in {"all", "preflight"}:
            run_preflight(args, root, split_file)
        if args.stage == "preflight":
            return
        if not split_file.is_file():
            raise FileNotFoundError(f"R2 frozen split missing: {split_file}")

        if args.stage in {"all", "standard"}:
            standard = run_standard(args, root, split_file)
        else:
            standard = require_selected(root, "standard")
        ensure_handoff(root / "standard_to_downstream_handoff.json", "standard", standard, split_file)
        validate_standard_gate(root, args.quick)
        if args.stage == "standard":
            return

        if args.stage in {"all", "teacher"}:
            anchor = run_teacher(args, root, split_file, standard)
        else:
            anchor = require_selected(root, "teacher_top2_at")
        ensure_handoff(root / "teacher_to_dual_handoff.json", "adversarial_anchor", anchor, split_file)
        validate_anchor_gate(root, args.quick)
        if args.stage == "teacher":
            return

        if args.stage in {"all", "dual"}:
            run_dual(args, root, split_file, standard, anchor)
        if args.stage == "dual":
            return
        if args.stage in {"all", "audit"}:
            run_audit(args, root)


if __name__ == "__main__":
    main()
