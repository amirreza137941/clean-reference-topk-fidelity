#!/usr/bin/env python3
"""Run robust and noise-only ImageNet R4 search cells."""

import argparse

from topk_imagenet_attack_pilot_r4_launcher import run_search_group
import topk_imagenet_attack_pilot_r4_common as common


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-self-test", action="store_true")
    args = parser.parse_args()
    run_search_group(common.LIGHT_METHODS, "light", run_tests=not args.skip_self_test)
