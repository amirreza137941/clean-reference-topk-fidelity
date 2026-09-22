#!/usr/bin/env python3
"""Incremental launcher for CIFAR-100 actual-defense calibration R1."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATOR = PROJECT_DIR / "calibrate_topk_cifar100_actual_defense_r1.py"
AUDITOR = PROJECT_DIR / "audit_topk_cifar100_actual_defense_calibration_r1.py"
SELF_TEST = PROJECT_DIR / "test_topk_cifar100_actual_defense_calibration_r1.py"
DEFAULT_DATA = Path("./datasets/cifar-100-python")
DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_OUTPUT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_calibration_train2k_r1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-100 defense calibration R1")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--r2-root", type=Path, default=DEFAULT_R2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--mc-chunk", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def validate_files() -> None:
    required_new = [
        EVALUATOR,
        AUDITOR,
        SELF_TEST,
        PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
    ]
    required_r2 = [
        PROJECT_DIR / "topk_cifar100_r2_common.py",
        PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    ]
    missing = [str(path) for path in required_new + required_r2 if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n  " + "\n  ".join(missing))


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


@contextmanager
def exclusive_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".calibration.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another calibration process holds {lock_path}") from exc
        yield


def self_test_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(SELF_TEST),
        "--data-dir",
        str(args.data_dir),
        "--r2-root",
        str(args.r2_root),
        "--device",
        args.device,
    ]


def evaluator_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--data-dir",
        str(args.data_dir),
        "--r2-root",
        str(args.r2_root),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--mc-chunk",
        str(args.mc_chunk),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.amp:
        command.append("--amp")
    return command


def audit_command(args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-u", str(AUDITOR), "--output-dir", str(args.output_dir)]


def main() -> None:
    args = parse_args()
    validate_files()
    if args.audit_only and args.self_test_only:
        raise ValueError("--audit-only and --self-test-only are mutually exclusive")
    commands: list[list[str]] = []
    if args.audit_only:
        commands = [audit_command(args)]
    elif args.self_test_only:
        commands = [self_test_command(args)]
    else:
        if not args.skip_self_test:
            commands.append(self_test_command(args))
        commands.extend([evaluator_command(args), audit_command(args)])
    if args.print_command:
        for command in commands:
            print(" ".join(command))
        return

    output_dir = args.output_dir.expanduser().resolve()
    with exclusive_lock(output_dir):
        for command in commands:
            run(command)


if __name__ == "__main__":
    main()

