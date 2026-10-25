#!/usr/bin/env python3
"""Shared launch configuration for the V7 common-MC=1000 re-evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parent
REEVALUATOR = PROJECT_DIR / "reevaluate_topk_v7_candidates_mc1000.py"
MODEL_ROOT = Path("./models/resnet20")

V7_FULL = MODEL_ROOT / "paper_topk_v7_hybrid_full_k2_q5k_seed3"
V7_NOISE = MODEL_ROOT / "paper_topk_v7_hybrid_noise_only_k2_q5k_seed3"
PRIOR_FULL = MODEL_ROOT / "paper_topk_candidate_reeval_full_k2_mc1000_r2"
PRIOR_NOISE = (
    MODEL_ROOT / "paper_topk_candidate_reeval_noise_only_k2_mc1000_r2"
)
OUTPUT_FULL = (
    MODEL_ROOT / "paper_topk_candidate_reeval_v5_v6_v7_full_k2_mc1000_r3"
)
OUTPUT_NOISE = (
    MODEL_ROOT / "paper_topk_candidate_reeval_v5_v6_v7_noise_only_k2_mc1000_r3"
)


def reeval_args(
    *,
    mode: str,
    v7_root: Path,
    prior_r2_dir: Path,
    output_dir: Path,
) -> list[str]:
    if mode not in {"full", "noise_only"}:
        raise ValueError(f"Unsupported mode: {mode}")
    batch_size = 5 if mode == "full" else 25
    return [
        "--mode",
        mode,
        "--v7-root",
        str(v7_root),
        "--prior-r2-dir",
        str(prior_r2_dir),
        "--output-dir",
        str(output_dir),
        "--eval-draws",
        "1000",
        "--reeval-seed",
        "2026082801",
        "--bootstrap-resamples",
        "50000",
        "--bootstrap-seed",
        "20260827",
        "--batch-size",
        str(batch_size),
        "--gpu-memory-fraction",
        "0.45",
        "--device",
        "cuda:0",
    ]


def validate_project_files() -> None:
    required = [
        REEVALUATOR,
        PROJECT_DIR / "reevaluate_topk_v5_v6_candidates_mc1000.py",
        PROJECT_DIR / "evaluate_general_topk_defense_v6_cifar10.py",
        PROJECT_DIR / "evaluate_final_top1_set2_defense_cifar10.py",
        PROJECT_DIR / "evaluate_top1_set2_gated_cifar10.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing project files:\n  " + "\n  ".join(missing))


def run_reevaluation(args: Sequence[str]) -> None:
    validate_project_files()
    subprocess.run(
        [sys.executable, "-u", str(REEVALUATOR), *args],
        cwd=PROJECT_DIR,
        check=True,
    )
