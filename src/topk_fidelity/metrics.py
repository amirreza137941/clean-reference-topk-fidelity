"""Metrics for clean-reference, unordered Top-k set fidelity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


def _validate_logits(logits: torch.Tensor, k: int) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if not 1 <= int(k) < logits.shape[1]:
        raise ValueError("k must satisfy 1 <= k < number of classes")


def ranked_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Return ranked class indices with shape ``[batch, k]``."""

    _validate_logits(logits, k)
    return logits.topk(int(k), dim=1, largest=True, sorted=True).indices


def sorted_set(indices: torch.Tensor) -> torch.Tensor:
    """Canonicalize a batch of index sets while discarding rank order."""

    if indices.ndim != 2:
        raise ValueError("indices must have shape [batch, k]")
    return indices.sort(dim=1).values


def exact_set_match(reference: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Return one when two unordered Top-k sets are exactly equal."""

    if reference.shape != predicted.shape or reference.ndim != 2:
        raise ValueError("reference and predicted must share shape [batch, k]")
    return sorted_set(reference).eq(sorted_set(predicted)).all(dim=1)


def set_membership(reference: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Mark which ranked reference items occur in the predicted set."""

    if reference.shape != predicted.shape or reference.ndim != 2:
        raise ValueError("reference and predicted must share shape [batch, k]")
    return reference[:, :, None].eq(predicted[:, None, :]).any(dim=2)


def boundary_margin(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Return the logit gap between positions k and k+1."""

    _validate_logits(logits, k)
    values = logits.topk(int(k) + 1, dim=1, largest=True, sorted=True).values
    return values[:, int(k) - 1] - values[:, int(k)]


def rank_weights(k: int, tau: float, *, device: torch.device) -> torch.Tensor:
    if int(k) < 1:
        raise ValueError("k must be positive")
    if float(tau) < 0:
        raise ValueError("tau must be non-negative")
    positions = torch.arange(int(k), device=device, dtype=torch.float32)
    weights = torch.exp(-float(tau) * positions)
    return weights / weights.sum()


@dataclass(frozen=True)
class MetricBatch:
    """Per-example values and their aggregate summary."""

    summary: Dict[str, float]
    values: Dict[str, torch.Tensor]
    reference_set: torch.Tensor
    predicted_set: torch.Tensor


def compute_metric_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    k: int,
    tau: float = 0.5,
) -> MetricBatch:
    """Compute the paper's clean-reference fidelity metrics.

    Percentage-valued entries in ``summary`` are on a 0--100 scale. The
    tensors in ``values`` stay on a 0--1 scale. ``coverage`` means that the
    ground-truth class is contained in the predicted Top-k set; it is distinct
    from exact clean-reference set fidelity.
    """

    _validate_logits(logits, k)
    _validate_logits(reference_logits, k)
    if logits.shape != reference_logits.shape:
        raise ValueError("logits and reference_logits must have the same shape")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")

    reference = ranked_topk(reference_logits, k)
    predicted = ranked_topk(logits, k)
    membership = set_membership(reference, predicted)
    intersection = membership.float().sum(dim=1)
    exact = exact_set_match(reference, predicted).float()
    coverage = predicted.eq(labels[:, None]).any(dim=1).float()
    top1 = logits.argmax(dim=1).eq(labels).float()
    overlap = intersection / float(k)
    jaccard = intersection / (2.0 * float(k) - intersection).clamp_min(1.0)
    weights = rank_weights(k, tau, device=logits.device)
    weighted_membership = (membership.float() * weights[None, :]).sum(dim=1)
    margin = boundary_margin(logits, k)
    reference_margin = boundary_margin(reference_logits, k)
    reference_top1_match = logits.argmax(dim=1).eq(reference[:, 0]).float()
    hidden_topk_failure = reference_top1_match * (1.0 - exact)

    values = {
        "top1": top1,
        "coverage": coverage,
        "exact_set": exact,
        "overlap": overlap,
        "jaccard": jaccard,
        "weighted_membership": weighted_membership,
        "margin": margin,
        "reference_margin": reference_margin,
        "reference_top1_match": reference_top1_match,
        "hidden_topk_failure": hidden_topk_failure,
    }
    percentages = {
        "top1",
        "coverage",
        "exact_set",
        "overlap",
        "jaccard",
        "weighted_membership",
        "reference_top1_match",
        "hidden_topk_failure",
    }
    summary = {
        name: (100.0 if name in percentages else 1.0) * float(value.mean().item())
        for name, value in values.items()
    }
    return MetricBatch(summary, values, reference, predicted)


def evaluate_draws(
    draw_logits: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    k: int,
    tau: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Average per-request metrics across stochastic defense draws.

    ``draw_logits`` must have shape ``[draws, batch, classes]``. This function
    deliberately evaluates every response separately before averaging; it does
    not compute metrics from mean logits.
    """

    if draw_logits.ndim != 3 or draw_logits.shape[0] < 1:
        raise ValueError("draw_logits must have shape [draws, batch, classes]")
    batches = [
        compute_metric_batch(draw, labels, reference_logits, k=k, tau=tau)
        for draw in draw_logits
    ]
    return {
        name: torch.stack([batch.values[name] for batch in batches], dim=0).mean(0)
        for name in batches[0].values
    }
