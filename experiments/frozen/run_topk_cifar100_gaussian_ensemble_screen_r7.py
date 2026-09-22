#!/usr/bin/env python3
"""Small preset launcher for the CIFAR-100 Gaussian-ensemble screening."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaussian-ensemble R7 preset launcher")
    parser.add_argument("--preset", choices=("latency", "quick", "full"), default="quick")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--r5-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args, extra = parser.parse_known_args()

    script = Path(__file__).resolve().with_name(
        "evaluate_topk_cifar100_gaussian_ensemble_screen_r7.py"
    )
    command = [
        sys.executable,
        "-u",
        str(script),
        "--project-dir",
        str(args.project_dir.expanduser().resolve()),
        "--device",
        args.device,
    ]
    if args.r5_root is not None:
        command.extend(["--r5-root", str(args.r5_root.expanduser().resolve())])
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir.expanduser().resolve())])

    if args.preset == "latency":
        command.extend(
            [
                "--stage",
                "latency",
                "--candidate-seeds",
                "71071",
                "--latency-warmup",
                "20",
                "--latency-repeats",
                "100",
            ]
        )
    elif args.preset == "quick":
        command.extend(
            [
                "--stage",
                "all",
                "--candidate-seeds",
                "71071",
                "--examples",
                "200",
                "--outer-draws",
                "20",
                "--latency-warmup",
                "10",
                "--latency-repeats",
                "50",
            ]
        )
    else:
        command.extend(
            [
                "--stage",
                "all",
                "--candidate-seeds",
                "71071,82073,93083",
                "--examples",
                "200",
                "--outer-draws",
                "100",
                "--latency-warmup",
                "20",
                "--latency-repeats",
                "100",
            ]
        )

    command.extend(extra)
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
