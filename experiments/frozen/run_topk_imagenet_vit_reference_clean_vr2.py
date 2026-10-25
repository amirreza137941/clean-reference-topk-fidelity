#!/usr/bin/env python3
"""Locked launcher for ImageNet ViT-reference clean confirmation VR2."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import subprocess
import sys
from pathlib import Path

import topk_imagenet_vit_reference_clean_vr2_common as common


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = Path(sys.executable).resolve()
PARENT_AUDITOR = PROJECT_DIR / "audit_topk_imagenet_vit_reference_calibration_vr1.py"
SELF_TEST = PROJECT_DIR / "test_topk_imagenet_vit_reference_clean_vr2.py"
EVALUATOR = PROJECT_DIR / "confirm_topk_imagenet_vit_reference_clean_vr2.py"
AUDITOR = PROJECT_DIR / "audit_topk_imagenet_vit_reference_clean_vr2.py"
PACKER = PROJECT_DIR / "pack_topk_imagenet_vit_reference_clean_vr2.py"

DEFAULT_ROOT = Path(".")
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "datasets/imagenet"
DEFAULT_R0_DIR = DEFAULT_ROOT / "models/imagenet_topk/paper_topk_imagenet_preflight_r0"
DEFAULT_R1_DIR = DEFAULT_ROOT / "models/imagenet_topk/paper_topk_imagenet_robust_anchor_r1"
DEFAULT_CALIBRATION_DIR = (
    DEFAULT_ROOT
    / "models/imagenet_topk/paper_topk_imagenet_vit_reference_crossarch_calibration_dev_vr1"
)
DEFAULT_STANDARD_CHECKPOINT = DEFAULT_ROOT / "models/vit/imagenet/vit_b16_imagenet.pth"
DEFAULT_ROBUST_CHECKPOINT = (
    DEFAULT_ROOT / "models/resnet50/imagenet/salman2020do_r50_linf4.pth"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ROOT
    / "models/imagenet_topk/paper_topk_imagenet_vit_reference_crossarch_clean_confirmation_vr2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked VR2 clean confirmation")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0_DIR)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--standard-checkpoint", type=Path, default=DEFAULT_STANDARD_CHECKPOINT)
    parser.add_argument("--robust-checkpoint", type=Path, default=DEFAULT_ROBUST_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mc-chunk", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def validate_files() -> None:
    required = {
        PARENT_AUDITOR,
        SELF_TEST,
        EVALUATOR,
        AUDITOR,
        PACKER,
        PROJECT_DIR / "topk_imagenet_vit_reference_clean_vr2_common.py",
        PROJECT_DIR / "topk_imagenet_vit_reference_calibration_vr1_common.py",
        PROJECT_DIR / "calibrate_topk_imagenet_vit_reference_vr1.py",
        PROJECT_DIR / "test_topk_imagenet_vit_reference_calibration_vr1.py",
        PROJECT_DIR / "run_topk_imagenet_vit_reference_calibration_vr1.py",
        PROJECT_DIR / "topk_imagenet_preflight_common.py",
        PROJECT_DIR / "preflight_topk_imagenet_r0.py",
        PROJECT_DIR / "topk_imagenet_robust_anchor_common_r1.py",
        PROJECT_DIR / "prepare_topk_imagenet_robust_anchor_r1.py",
    }
    missing = [str(path) for path in sorted(required) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source files are missing:\n" + "\n".join(missing))
    for filename, expected in common.EXPECTED_CALIBRATION_SOURCE_HASHES.items():
        path = PROJECT_DIR / filename
        if common.file_sha256(path) != expected:
            raise AssertionError(f"Frozen calibration source hash mismatch: {filename}")


def shared_path_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root", str(args.data_root),
        "--r0-dir", str(args.r0_dir),
        "--r1-dir", str(args.r1_dir),
        "--calibration-dir", str(args.calibration_dir),
        "--standard-checkpoint", str(args.standard_checkpoint),
        "--robust-checkpoint", str(args.robust_checkpoint),
    ]


def parent_audit_command(args: argparse.Namespace) -> list[str]:
    return [str(PYTHON), "-u", str(PARENT_AUDITOR), "--output-dir", str(args.calibration_dir)]


def self_test_command(args: argparse.Namespace) -> list[str]:
    return [
        str(PYTHON), "-u", str(SELF_TEST),
        *shared_path_args(args),
        "--device", str(args.device),
        "--gpu-memory-fraction", str(args.gpu_memory_fraction),
    ]


def evaluator_command(args: argparse.Namespace) -> list[str]:
    return [
        str(PYTHON), "-u", str(EVALUATOR),
        *shared_path_args(args),
        "--output-dir", str(args.output_dir),
        "--device", str(args.device),
        "--batch-size", str(args.batch_size),
        "--mc-chunk", str(args.mc_chunk),
        "--num-workers", str(args.num_workers),
        "--gpu-memory-fraction", str(args.gpu_memory_fraction),
    ]


def audit_command(args: argparse.Namespace) -> list[str]:
    return [str(PYTHON), "-u", str(AUDITOR), "--output-dir", str(args.output_dir)]


def pack_command(args: argparse.Namespace) -> list[str]:
    command = [str(PYTHON), "-u", str(PACKER), "--output-dir", str(args.output_dir)]
    if args.archive is not None:
        command.extend(["--archive", str(args.archive)])
    return command


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


@contextlib.contextmanager
def exclusive_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".vit_reference_clean_vr2.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another VR2 process holds {lock_path}") from exc
        yield


def main() -> None:
    args = parse_args()
    if args.audit_only and args.self_test_only:
        raise ValueError("--audit-only and --self-test-only are mutually exclusive")
    validate_files()
    commands: list[list[str]] = [parent_audit_command(args)]
    if args.self_test_only:
        commands.append(self_test_command(args))
    elif args.audit_only:
        commands.append(audit_command(args))
        if not args.skip_pack:
            commands.append(pack_command(args))
    else:
        if not args.skip_self_test:
            commands.append(self_test_command(args))
        commands.extend([evaluator_command(args), audit_command(args)])
        if not args.skip_pack:
            commands.append(pack_command(args))
    if args.print_command:
        for command in commands:
            print(" ".join(command))
        return
    print(f"sys.executable={PYTHON}", flush=True)
    print(f"write target={args.output_dir.expanduser().resolve()}", flush=True)
    with exclusive_lock(args.output_dir.expanduser().resolve()):
        for command in commands:
            run(command)


if __name__ == "__main__":
    main()
