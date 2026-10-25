#!/usr/bin/env python3
"""One-command launcher for ImageNet Top-k R0 preflight and audit."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("./datasets/imagenet")
DEFAULT_RESNET50 = Path(
    "./models/resnet50/imagenet/"
    "resnet50_imagenet.pth"
)
DEFAULT_VIT_B16 = Path(
    "./models/vit/imagenet/"
    "vit_b16_imagenet.pth"
)
DEFAULT_OUTPUT = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_preflight_r0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ImageNet Top-k R0 preflight")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--resnet50-checkpoint", type=Path, default=DEFAULT_RESNET50)
    parser.add_argument("--vit-b16-checkpoint", type=Path, default=DEFAULT_VIT_B16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-seed", type=int, default=9102026)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--memory-batch-sizes", default="1,2,4,8,16,24,32")
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--skip-memory-sweep", action="store_true")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)


def main() -> None:
    args = parse_args()
    required = [
        "topk_imagenet_preflight_common.py",
        "preflight_topk_imagenet_r0.py",
        "audit_topk_imagenet_preflight_r0.py",
        "test_topk_imagenet_preflight_r0.py",
    ]
    for name in required:
        path = PROJECT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"sys.executable={sys.executable}")
    print(f"project_dir={PROJECT_DIR}")
    print(f"write target={args.output_dir.expanduser().resolve()}")
    print("This launcher is offline: no download, training, tuning, or attack.")
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    common_paths = [
        "--data-root",
        str(args.data_root),
        "--resnet50-checkpoint",
        str(args.resnet50_checkpoint),
        "--vit-b16-checkpoint",
        str(args.vit_b16_checkpoint),
        "--device",
        args.device,
    ]
    if not args.skip_self_test:
        run(
            [sys.executable, "-u", str(PROJECT_DIR / "test_topk_imagenet_preflight_r0.py")]
            + common_paths,
            env,
        )

    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "preflight_topk_imagenet_r0.py"),
        *common_paths,
        "--output-dir",
        str(args.output_dir),
        "--split-seed",
        str(args.split_seed),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
        "--memory-batch-sizes",
        args.memory_batch_sizes,
    ]
    if args.skip_memory_sweep:
        command.append("--skip-memory-sweep")
    if args.allow_unsafe_checkpoint_load:
        command.append("--allow-unsafe-checkpoint-load")
    run(command, env)
    run(
        [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "audit_topk_imagenet_preflight_r0.py"),
            "--output-dir",
            str(args.output_dir),
        ],
        env,
    )


if __name__ == "__main__":
    main()

