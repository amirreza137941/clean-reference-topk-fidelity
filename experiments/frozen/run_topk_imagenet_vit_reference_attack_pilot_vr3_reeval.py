#!/usr/bin/env python3
"""Run common-MC=1000 reevaluation and the final read-only VR3 audit."""

import argparse

from topk_imagenet_vit_reference_attack_pilot_vr3_launcher import run_reevaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-self-test", action="store_true")
    args = parser.parse_args()
    run_reevaluation(run_tests=not args.skip_self_test)
