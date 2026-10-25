#!/usr/bin/env python3
"""Run/resume frozen Robust Q=20k D1 search and reevaluation."""

from topk_imagenet_posthoc_q20k_eot_d1_launcher import run_method


if __name__ == "__main__":
    run_method(method="robust")

