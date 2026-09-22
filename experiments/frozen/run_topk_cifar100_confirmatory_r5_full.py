#!/usr/bin/env python3
"""Run/resume frozen Full/SquareAdaptive R5 search and reevaluation."""

from topk_cifar100_confirmatory_r5_launcher import run_method


if __name__ == "__main__":
    run_method(method="full")

