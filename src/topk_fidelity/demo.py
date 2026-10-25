"""A small deterministic CPU example of the full defense/search/evaluation path."""

from __future__ import annotations

import torch
from torch import nn

from .attacks import run_square_batch
from .config import AdaptiveConfig, DefenseConfig, RacingConfig
from .defense import CleanReferenceTopKDefense
from .evaluation import evaluate_responses
from .metrics import ranked_topk
from .models import freeze_model


def run_demo(attack: str = "SquareAdaptive", seed: int = 7) -> dict[str, object]:
    """Exercise the actual reusable algorithms; these are not paper results."""
    torch.manual_seed(seed)
    reference = freeze_model(nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6)))
    anchor = freeze_model(nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6)))
    images = torch.rand(2, 3, 8, 8)
    labels = torch.tensor([0, 1])
    defense = CleanReferenceTopKDefense(
        reference,
        anchor,
        anchor,
        DefenseConfig(top_k=2, alpha=0.005, epsilon=0.03, steps=2, sigma=0.01),
    )
    with torch.no_grad():
        clean_logits = reference(images)
    candidate, details = run_square_batch(
        attack=attack,
        defense=defense,
        clean=images,
        reference_topk=ranked_topk(clean_logits, 2),
        eps=2 / 255,
        api_budget=64,
        p_init=0.30,
        adaptive_config=AdaptiveConfig().as_dict(),
        racing_config=RacingConfig(race_interval=2, pool_size=2).as_dict(),
        proposal_seed=seed + 1,
        oracle_seed=seed + 2,
    )
    values = evaluate_responses(
        defense,
        candidate,
        labels,
        clean_logits,
        k=2,
        draws=7,
        chunk_size=3,
    )
    return {
        "purpose": "CPU example; synthetic models and data, not paper evidence",
        "attack": attack,
        "examples": len(images),
        "api_queries_per_image": details["api_queries"],
        "endpoint_draws_outside_attack_budget": 7,
        "maximum_attack_linf": float((candidate - images).abs().max()),
        "attacked_metrics_pct": {
            name: 100 * float(value.mean())
            for name, value in values.items()
            if "margin" not in name
        },
        "attack_details": details,
    }
