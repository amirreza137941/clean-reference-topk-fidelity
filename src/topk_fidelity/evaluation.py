"""Endpoint evaluation using fixed clean targets and per-response metrics."""

from __future__ import annotations

from typing import Any

import torch

from .config import positive_integer
from .metrics import compute_metric_batch
from .query import collect_draw_logits


def evaluate_responses(
    defense: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_reference_logits: torch.Tensor,
    *,
    k: int,
    draws: int,
    chunk_size: int = 20,
) -> dict[str, torch.Tensor]:
    """Average per-draw metrics with bounded response memory.

    Use only after candidate search and confirmation finish. No attack ledger
    is charged and endpoint metrics must not feed back into candidate selection.
    Clean reference logits are fixed; the defense uses an online reference.

    Values are per-image probabilities except raw logit margins. chunk_size
    affects memory use, not request order or number of responses.
    """
    positive_integer(draws, "draws")
    positive_integer(chunk_size, "chunk_size")
    totals: dict[str, torch.Tensor] = {}
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        responses = collect_draw_logits(defense, images, count)
        for response in responses:
            values = compute_metric_batch(
                response, labels, clean_reference_logits, k=k
            ).values
            for name, value in values.items():
                contribution = value.double()
                totals[name] = (
                    totals.get(name, torch.zeros_like(contribution)) + contribution
                )
    return {name: value / draws for name, value in totals.items()}
