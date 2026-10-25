"""Original exact-set objectives; full class logits enter every margin."""

from __future__ import annotations

import torch


def exact_set_attack_margin(
    logits: torch.Tensor, reference_topk: torch.Tensor
) -> torch.Tensor:
    """Positive iff all reference items outrank every outside item (ties aside)."""

    selected = logits.gather(1, reference_topk)
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference_topk, False)
    strongest_outside = logits.masked_fill(~outside, float("-inf")).max(1).values
    return selected.min(1).values - strongest_outside


def per_draw_exact_set_survival_score(
    draw_logits: torch.Tensor,
    reference_topk: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Smooth Monte Carlo estimate of per-request exact-set survival."""

    if draw_logits.ndim != 3:
        raise ValueError("draw_logits must have shape [draws,batch,classes]")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    draws, batch, classes = draw_logits.shape
    if reference_topk.ndim != 2 or reference_topk.shape[0] != batch:
        raise ValueError("reference_topk must have shape [batch,k]")
    expanded = reference_topk.unsqueeze(0).expand(draws, -1, -1)
    selected = draw_logits.gather(2, expanded)
    outside = torch.ones(
        (draws, batch, classes), dtype=torch.bool, device=draw_logits.device
    )
    outside.scatter_(2, expanded, False)
    strongest_outside = draw_logits.masked_fill(~outside, float("-inf")).max(2).values
    margins = selected.min(2).values - strongest_outside
    return torch.sigmoid(margins / float(temperature)).mean(0)
