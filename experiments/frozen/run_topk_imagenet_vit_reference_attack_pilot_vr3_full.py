#!/usr/bin/env python3
"""Run full-sigma0 and full ImageNet ViT-reference VR3 search cells."""

import argparse

from topk_imagenet_vit_reference_attack_pilot_vr3_launcher import run_search_group
import topk_imagenet_vit_reference_attack_pilot_vr3_common as common


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-self-test", action="store_true")
    args = parser.parse_args()
    run_search_group(common.FULL_METHODS, "full", run_tests=not args.skip_self_test)
