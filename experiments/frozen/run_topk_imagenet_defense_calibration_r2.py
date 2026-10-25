#!/usr/bin/env python3
"""Locked, resumable launcher for ImageNet Top-k calibration R2."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATOR = PROJECT_DIR / "calibrate_topk_imagenet_defense_r2.py"
AUDITOR = PROJECT_DIR / "audit_topk_imagenet_defense_calibration_r2.py"
SELF_TEST = PROJECT_DIR / "test_topk_imagenet_defense_calibration_r2.py"
DEFAULT_DATA_ROOT = Path("./datasets/imagenet")
DEFAULT_R0_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_preflight_r0"
)
DEFAULT_R1_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_robust_anchor_r1"
)
DEFAULT_STANDARD_CHECKPOINT = Path(
    "./models/resnet50/imagenet/resnet50_imagenet.pth"
)
DEFAULT_ROBUST_CHECKPOINT = Path(
    "./models/resnet50/imagenet/"
    "salman2020do_r50_linf4.pth"
)
DEFAULT_OUTPUT_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_defense_calibration_dev_r2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ImageNet defense calibration R2")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0_DIR)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    parser.add_argument("--standard-checkpoint", type=Path, default=DEFAULT_STANDARD_CHECKPOINT)
    parser.add_argument("--robust-checkpoint", type=Path, default=DEFAULT_ROBUST_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mc-chunk", type=int, default=4)
    parser.add_argument("--sigma-forward-group", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def validate_files() -> None:
    names = (
        "topk_imagenet_defense_calibration_r2_common.py",
        "calibrate_topk_imagenet_defense_r2.py",
        "audit_topk_imagenet_defense_calibration_r2.py",
        "test_topk_imagenet_defense_calibration_r2.py",
        "run_topk_imagenet_defense_calibration_r2.py",
        "topk_imagenet_preflight_common.py",
        "preflight_topk_imagenet_r0.py",
        "topk_imagenet_robust_anchor_common_r1.py",
        "prepare_topk_imagenet_robust_anchor_r1.py",
    )
    missing = [str(PROJECT_DIR / name) for name in names if not (PROJECT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Required source files are missing:\n  " + "\n  ".join(missing))


def shared_path_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root", str(args.data_root),
        "--r0-dir", str(args.r0_dir),
        "--r1-dir", str(args.r1_dir),
        "--standard-checkpoint", str(args.standard_checkpoint),
        "--robust-checkpoint", str(args.robust_checkpoint),
        "--device", args.device,
        "--gpu-memory-fraction", str(args.gpu_memory_fraction),
    ]


def self_test_command(args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-u", str(SELF_TEST), *shared_path_args(args)]


def evaluator_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        *shared_path_args(args),
        "--output-dir", str(args.output_dir),
        "--batch-size", str(args.batch_size),
        "--mc-chunk", str(args.mc_chunk),
        "--sigma-forward-group", str(args.sigma_forward_group),
        "--num-workers", str(args.num_workers),
    ]
    if args.amp:
        command.append("--amp")
    return command


def audit_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(AUDITOR),
        "--output-dir",
        str(args.output_dir),
    ]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


@contextmanager
def exclusive_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".calibration_r2.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another R2 process holds {lock_path}") from exc
        yield


def main() -> None:
    args = parse_args()
    validate_files()
    if args.audit_only and args.self_test_only:
        raise ValueError("--audit-only and --self-test-only are mutually exclusive")
    if args.audit_only:
        commands = [audit_command(args)]
    elif args.self_test_only:
        commands = [self_test_command(args)]
    else:
        commands = []
        if not args.skip_self_test:
            commands.append(self_test_command(args))
        commands.extend((evaluator_command(args), audit_command(args)))
    if args.print_command:
        for command in commands:
            print(" ".join(command))
        return
    with exclusive_lock(args.output_dir.expanduser().resolve()):
        for command in commands:
            run(command)


if __name__ == "__main__":
    main()
