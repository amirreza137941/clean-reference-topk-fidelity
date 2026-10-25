"""Compatibility imports; implementation is split by responsibility."""

from .adaptive import square_adaptive_attack_budget
from .objectives import exact_set_attack_margin, per_draw_exact_set_survival_score
from .plans import (
    square_adaptive_query_plan,
    square_racing_query_plan,
    square_reference_p_selection,
)
from .proposals import (
    square_reference_linf_proposal,
    square_vertical_stripe_initialization,
)
from .racing import square_racing_attack_budget
from .runner import run_square_batch

__all__ = [
    "exact_set_attack_margin",
    "per_draw_exact_set_survival_score",
    "run_square_batch",
    "square_adaptive_attack_budget",
    "square_adaptive_query_plan",
    "square_racing_attack_budget",
    "square_racing_query_plan",
    "square_reference_linf_proposal",
    "square_reference_p_selection",
    "square_vertical_stripe_initialization",
]
