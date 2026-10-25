#!/usr/bin/env python3
"""Hash-locked bridge to the already-audited generic Square R4 core.

The implementation first shipped with the CIFAR-100 stage, but all of its
tensor operations are dataset agnostic.  Reusing the exact audited bytes
avoids maintaining a divergent ImageNet copy.  The dependency and its SHA256
are frozen in the ImageNet R4 protocol.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import topk_imagenet_attack_pilot_r4_common as common


def _load_core():
    path = common.DEFAULT_CIFAR_CORE.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing audited Square core: {path}\n"
            "Keep the previously delivered CIFAR100 R4 files in the declared directory."
        )
    observed = common.file_sha256(path)
    if observed != common.EXPECTED_ATTACK_CORE_SHA256:
        raise AssertionError(
            f"Square core SHA256 mismatch: expected {common.EXPECTED_ATTACK_CORE_SHA256}, "
            f"observed {observed}"
        )
    spec = importlib.util.spec_from_file_location("_audited_topk_square_core_r4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Square core from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CORE = _load_core()

stable_int = _CORE.stable_int
seed_everything = _CORE.seed_everything
ranked_topk = _CORE.ranked_topk
exact_set_attack_margin = _CORE.exact_set_attack_margin
per_draw_exact_set_survival_score = _CORE.per_draw_exact_set_survival_score
square_reference_p_selection = _CORE.square_reference_p_selection
square_adaptive_query_plan = _CORE.square_adaptive_query_plan
square_racing_query_plan = _CORE.square_racing_query_plan
square_vertical_stripe_initialization = _CORE.square_vertical_stripe_initialization
square_reference_linf_proposal = _CORE.square_reference_linf_proposal
run_square_batch = _CORE.run_square_batch

__all__ = [
    "stable_int",
    "seed_everything",
    "ranked_topk",
    "exact_set_attack_margin",
    "per_draw_exact_set_survival_score",
    "square_reference_p_selection",
    "square_adaptive_query_plan",
    "square_racing_query_plan",
    "square_vertical_stripe_initialization",
    "square_reference_linf_proposal",
    "run_square_batch",
]
