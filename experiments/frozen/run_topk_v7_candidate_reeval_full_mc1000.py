#!/usr/bin/env python3
"""Launch common-MC=1000 re-evaluation for the three Full V7 candidates."""

from topk_v7_candidate_reeval_launcher_common import (
    OUTPUT_FULL,
    PRIOR_FULL,
    V7_FULL,
    reeval_args,
    run_reevaluation,
)


if __name__ == "__main__":
    run_reevaluation(
        reeval_args(
            mode="full",
            v7_root=V7_FULL,
            prior_r2_dir=PRIOR_FULL,
            output_dir=OUTPUT_FULL,
        )
    )
