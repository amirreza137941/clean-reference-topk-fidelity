#!/usr/bin/env python3
"""Launch common-MC=1000 re-evaluation for three Noise-only V7 candidates."""

from topk_v7_candidate_reeval_launcher_common import (
    OUTPUT_NOISE,
    PRIOR_NOISE,
    V7_NOISE,
    reeval_args,
    run_reevaluation,
)


if __name__ == "__main__":
    run_reevaluation(
        reeval_args(
            mode="noise_only",
            v7_root=V7_NOISE,
            prior_r2_dir=PRIOR_NOISE,
            output_dir=OUTPUT_NOISE,
        )
    )
