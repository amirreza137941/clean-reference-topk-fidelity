#!/usr/bin/env python3
"""Launcher for frozen CIFAR-100 official clean confirmation R3."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATOR = PROJECT_DIR / "evaluate_topk_cifar100_official_clean_r3.py"
AUDITOR = PROJECT_DIR / "audit_topk_cifar100_official_clean_r3.py"
SELF_TEST = PROJECT_DIR / "test_topk_cifar100_official_clean_r3.py"
DEFAULT_DATA = Path("./datasets/cifar-100-python")
DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_DEV_DIR = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_dev_pareto_r2"
)
DEFAULT_OUTPUT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_official_clean_confirmation_k2_mc100_r3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-100 official clean R3")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--r2-root", type=Path, default=DEFAULT_R2_ROOT)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def validate_files() -> None:
    required = [
        EVALUATOR,
        AUDITOR,
        SELF_TEST,
        PROJECT_DIR / "topk_cifar100_official_clean_r3_common.py",
        PROJECT_DIR / "calibrate_topk_cifar100_dev_pareto_r2.py",
        PROJECT_DIR / "topk_cifar100_dev_pareto_r2_common.py",
        PROJECT_DIR / "calibrate_topk_cifar100_actual_defense_r1.py",
        PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
        PROJECT_DIR / "topk_cifar100_r2_common.py",
        PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n  " + "\n  ".join(missing))


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


@contextmanager
def exclusive_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".official_clean_r3.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another official-clean process holds {lock_path}") from exc
        yield


def shared_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-dir",
        str(args.data_dir),
        "--r2-root",
        str(args.r2_root),
        "--dev-dir",
        str(args.dev_dir),
        "--device",
        args.device,
    ]


def self_test_command(args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-u", str(SELF_TEST), *shared_args(args)]


def evaluator_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(EVALUATOR),
        *shared_args(args),
        "--output-dir",
        str(args.output_dir),
        "--num-workers",
        str(args.num_workers),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
    ]


def audit_command(args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-u", str(AUDITOR), "--output-dir", str(args.output_dir)]


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
