#!/usr/bin/env python3
"""Single-command launcher for ImageNet R3 self-test, run, and audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_clean_confirmation_k5_mc50_r3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ImageNet R3 clean confirmation")
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    shared = [
        "--device",
        args.device,
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
    ]
    if not args.skip_self_test:
        run(
            [
                python,
                "-u",
                str(PROJECT_DIR / "test_topk_imagenet_clean_confirmation_r3.py"),
                *shared,
            ]
        )
    run(
        [
            python,
            "-u",
            str(PROJECT_DIR / "evaluate_topk_imagenet_clean_confirmation_r3.py"),
            "--output-dir",
            str(args.output_dir.expanduser().resolve()),
            "--num-workers",
            str(args.num_workers),
            *shared,
        ]
    )
    run(
        [
            python,
            "-u",
            str(PROJECT_DIR / "audit_topk_imagenet_clean_confirmation_r3.py"),
            "--output-dir",
            str(args.output_dir.expanduser().resolve()),
        ]
    )


if __name__ == "__main__":
    main()
