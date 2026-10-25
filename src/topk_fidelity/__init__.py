"""Original defense and attack utilities for clean-reference Top-k fidelity."""

from .config import AdaptiveConfig, DefenseConfig, ExperimentConfig, RacingConfig
from .defense import CleanReferenceTopKDefense
from .evaluation import evaluate_responses
from .metrics import (
    MetricBatch,
    boundary_margin,
    compute_metric_batch,
    exact_set_match,
    ranked_topk,
)
from .query import AveragedQueryOracle, PerDrawQueryOracle, QueryLedger

__all__ = [
    "AdaptiveConfig",
    "AveragedQueryOracle",
    "CleanReferenceTopKDefense",
    "DefenseConfig",
    "ExperimentConfig",
    "MetricBatch",
    "PerDrawQueryOracle",
    "QueryLedger",
    "RacingConfig",
    "boundary_margin",
    "compute_metric_batch",
    "evaluate_responses",
    "exact_set_match",
    "ranked_topk",
]

__version__ = "0.2.0"
