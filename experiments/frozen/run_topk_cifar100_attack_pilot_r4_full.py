#!/usr/bin/env python3
"""Run/resume full-sigma0 and full-defense R4 search cells."""

import topk_cifar100_attack_pilot_r4_common as common
from topk_cifar100_attack_pilot_r4_launcher import run_group


if __name__ == "__main__":
    run_group(
        group="full",
        methods=common.FULL_METHODS,
        batch_size=common.SEARCH_BATCH_SIZE_BY_METHOD["full"],
    )
