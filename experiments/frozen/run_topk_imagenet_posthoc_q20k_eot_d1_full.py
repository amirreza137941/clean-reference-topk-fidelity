#!/usr/bin/env python3
"""Run/resume frozen Full Q=20k/EOT=5 D1 search and MC1000 reevaluation."""

from topk_imagenet_posthoc_q20k_eot_d1_launcher import run_method


if __name__ == "__main__":
    run_method(method="full")

