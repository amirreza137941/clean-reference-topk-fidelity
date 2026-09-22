"""Clean-reference Top-k fidelity utilities."""

from .defense import CleanReferenceTopKDefense, DefenseConfig
from .metrics import (
    MetricBatch,
    boundary_margin,
    compute_metric_batch,
    exact_set_match,
    ranked_topk,
)
from .query import AveragedQueryOracle, PerDrawQueryOracle, QueryLedger

__all__ = [
    "AveragedQueryOracle",
    "CleanReferenceTopKDefense",
    "DefenseConfig",
    "MetricBatch",
    "PerDrawQueryOracle",
    "QueryLedger",
    "boundary_margin",
    "compute_metric_batch",
    "exact_set_match",
    "ranked_topk",
]

__version__ = "0.1.0"
