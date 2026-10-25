#!/usr/bin/env python3
"""Run common-MC=1000 reevaluation and the final read-only R4 audit."""

import argparse

from topk_imagenet_attack_pilot_r4_launcher import run_reevaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-self-test", action="store_true")
    args = parser.parse_args()
    run_reevaluation(run_tests=not args.skip_self_test)
