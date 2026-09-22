"""Attack utilities with explicit query-budget accounting."""

from .square import (
    exact_set_attack_margin,
    square_adaptive_query_plan,
    square_racing_query_plan,
    square_reference_p_selection,
)

__all__ = [
    "exact_set_attack_margin",
    "square_adaptive_query_plan",
    "square_racing_query_plan",
    "square_reference_p_selection",
]
