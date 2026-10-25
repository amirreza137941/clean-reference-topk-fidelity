"""Original refresh-and-confirm SquareAdaptive search loop."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ..query import AveragedQueryOracle
from .archive import _archive_accepted_candidates
from .plans import square_adaptive_query_plan, square_reference_p_selection
from .proposals import (
    square_reference_linf_proposal,
    square_vertical_stripe_initialization,
)


def square_adaptive_attack_budget(
    oracle: AveragedQueryOracle,
    x: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float,
    api_budget: int,
    p_init: float,
    search_eot: int,
    refresh_eot: int,
    confirm_candidates: int,
    confirm_fraction: float,
    stochastic_defense: bool,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Run refresh-and-confirm Square Attack under one exact API budget.

    Unlike a stale noisy incumbent score, every proposal is compared against a
    fresh incumbent estimate.  A per-example archive is then re-scored with
    independent draws, and the confirmed minimum is returned.  The function
    always consumes exactly ``api_budget`` returned logit vectors per example.
    """

    api_budget = int(api_budget)
    search_eot = int(search_eot)
    refresh_eot = int(refresh_eot)
    pool_size = int(confirm_candidates)
    plan = square_adaptive_query_plan(
        api_budget=api_budget,
        search_eot=search_eot,
        refresh_eot=refresh_eot,
        confirm_candidates=pool_size,
        confirm_fraction=confirm_fraction,
        stochastic_defense=stochastic_defense,
    )
    search_steps = int(plan["search_steps"])

    incumbent = square_vertical_stripe_initialization(x, eps, rng)
    incumbent_score = score_fn(oracle.query(incumbent, search_eot))
    archive = incumbent.unsqueeze(0).repeat(pool_size, *([1] * incumbent.ndim))
    archive_scores = torch.full(
        (pool_size, len(x)),
        float("inf"),
        device=x.device,
        dtype=incumbent_score.dtype,
    )
    archive_scores[0] = incumbent_score

    for iteration in range(search_steps):
        p = square_reference_p_selection(p_init, iteration, max(search_steps, 1))
        candidate = square_reference_linf_proposal(x, incumbent, eps, p, rng)
        candidate_score = score_fn(oracle.query(candidate, search_eot))
        if refresh_eot > 0:
            fresh_incumbent_score = score_fn(oracle.query(incumbent, refresh_eot))
        else:
            fresh_incumbent_score = incumbent_score
        accepted = candidate_score < fresh_incumbent_score
        incumbent[accepted] = candidate[accepted]
        incumbent_score = torch.where(accepted, candidate_score, fresh_incumbent_score)
        _archive_accepted_candidates(
            archive,
            archive_scores,
            candidate,
            candidate_score,
            accepted,
        )

    # Guarantee that the terminal incumbent participates in confirmation even
    # if noisy archive ranking would otherwise have evicted it.
    archive[-1] = incumbent
    archive_scores[-1] = incumbent_score
    search_queries = oracle.api_queries_per_example
    if search_queries != int(plan["search_queries"]):
        raise RuntimeError("SquareAdaptive search-query accounting mismatch")
    confirmation_queries = int(plan["confirmation_queries"])
    draw_allocation = [int(value) for value in plan["confirmation_draws"]]
    confirmed_scores = []
    for slot, draws in enumerate(draw_allocation):
        confirmed_scores.append(score_fn(oracle.query(archive[slot], draws)))
    score_matrix = torch.stack(confirmed_scores, dim=0)
    best_slots = score_matrix.argmin(dim=0)
    samples = torch.arange(len(x), device=x.device)
    confirmed = archive[best_slots, samples].detach().clone()

    if oracle.api_queries_per_example != api_budget:
        raise RuntimeError(
            "SquareAdaptive query accounting mismatch: "
            f"used {oracle.api_queries_per_example}, expected {api_budget}"
        )
    details = {
        "search_steps": int(search_steps),
        "search_eot": int(search_eot),
        "refresh_eot": int(refresh_eot),
        "confirmation_candidates": int(pool_size),
        "confirmation_queries": int(confirmation_queries),
        "confirmation_draws_min": int(min(draw_allocation)),
        "confirmation_draws_max": int(max(draw_allocation)),
        "logical_calls": int(oracle.logical_calls),
        "api_queries": int(oracle.api_queries_per_example),
    }
    return confirmed, details
