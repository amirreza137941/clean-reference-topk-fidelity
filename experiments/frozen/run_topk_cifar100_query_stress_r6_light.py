#!/usr/bin/env python3
"""Run/resume Robust and Noise-only R6 cells sequentially."""

from topk_cifar100_query_stress_r6_launcher import run_group


if __name__ == "__main__":
    run_group(("robust", "noise_only"), label="light")

