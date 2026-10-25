#!/usr/bin/env python3
"""Launcher for ImageNet robust-anchor R1: self-test, prepare, audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_topk_imagenet_robust_anchor_r1 import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_R0_DIR,
    DEFAULT_RAW_CHECKPOINT,
    DEFAULT_STANDARD_CHECKPOINT,
    DEFAULT_EXPORT_CHECKPOINT,
)


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ImageNet robust-anchor R1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0_DIR)
    parser.add_argument(
        "--standard-checkpoint", type=Path, default=DEFAULT_STANDARD_CHECKPOINT
    )
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW_CHECKPOINT)
    parser.add_argument(
        "--export-checkpoint", type=Path, default=DEFAULT_EXPORT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--download-backend", choices=("auto", "gdown", "urllib"), default="auto"
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--pgd-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--skip-self-test", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    args = parse_args()
    print(f"sys.executable={sys.executable}")
    print(f"project_dir={PROJECT_DIR}")
    print(f"write target={args.output_dir.expanduser().resolve()}")
    print(f"raw checkpoint target={args.raw_checkpoint.expanduser().resolve()}")
    print("Network scope: one RobustBench checkpoint only; ImageNet is not downloaded.")

    if not args.skip_self_test:
        run(
            [
                sys.executable,
                "-u",
                str(PROJECT_DIR / "test_topk_imagenet_robust_anchor_r1.py"),
                "--project-dir",
                str(PROJECT_DIR),
                "--r0-dir",
                str(args.r0_dir),
                "--standard-checkpoint",
                str(args.standard_checkpoint),
            ]
        )

    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "prepare_topk_imagenet_robust_anchor_r1.py"),
        "--project-dir",
        str(PROJECT_DIR),
        "--data-root",
        str(args.data_root),
        "--r0-dir",
        str(args.r0_dir),
        "--standard-checkpoint",
        str(args.standard_checkpoint),
        "--raw-checkpoint",
        str(args.raw_checkpoint),
        "--export-checkpoint",
        str(args.export_checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--download-backend",
        args.download_backend,
        "--device",
        args.device,
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--pgd-batch-size",
        str(args.pgd_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
    ]
    if args.no_download:
        command.append("--no-download")
    run(command)
    run(
        [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "audit_topk_imagenet_robust_anchor_r1.py"),
            "--output-dir",
            str(args.output_dir),
        ]
    )


if __name__ == "__main__":
    main()
