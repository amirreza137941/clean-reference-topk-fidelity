"""Original per-draw SquareRacing search loop and archive refresh."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ..query import PerDrawQueryOracle
from .archive import _insert_racing_candidate
from .plans import square_racing_query_plan, square_reference_p_selection
from .proposals import (
    square_reference_linf_proposal,
    square_vertical_stripe_initialization,
)


def square_racing_attack_budget(
    oracle: PerDrawQueryOracle,
    x: torch.Tensor,
    score_draws_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float,
    api_budget: int,
    p_init: float,
    search_eot: int,
    refresh_eot: int,
    pool_size: int,
    race_interval: int,
    race_eot: int,
    confirm_fraction: float,
    stochastic_defense: bool,
    score_temperature: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run per-draw Square search with periodic archive racing.

    Candidate discovery uses independent multi-draw scores, the incumbent is
    freshly re-scored before every stochastic comparison, and every retained
    archive member is periodically refreshed.  Final selection uses a disjoint
    confirmation allocation.  Every returned logit vector is charged.
    """

    plan = square_racing_query_plan(
        api_budget=api_budget,
        search_eot=search_eot,
        refresh_eot=refresh_eot,
        pool_size=pool_size,
        race_interval=race_interval,
        race_eot=race_eot,
        confirm_fraction=confirm_fraction,
        stochastic_defense=stochastic_defense,
    )
    search_steps = int(plan["search_steps"])
    planned_races = int(plan["race_rounds"])

    incumbent = square_vertical_stripe_initialization(x, eps, rng)
    incumbent_score = score_draws_fn(oracle.query(incumbent, search_eot))
    archive = incumbent.unsqueeze(0).repeat(pool_size, *([1] * incumbent.ndim))
    archive_scores = incumbent_score.unsqueeze(0).repeat(pool_size, 1)
    filled = 1
    races_run = 0

    for iteration in range(search_steps):
        p = square_reference_p_selection(p_init, iteration, max(search_steps, 1))
        candidate = square_reference_linf_proposal(x, incumbent, eps, p, rng)
        candidate_score = score_draws_fn(oracle.query(candidate, search_eot))
        if refresh_eot > 0:
            fresh_incumbent_score = score_draws_fn(oracle.query(incumbent, refresh_eot))
        else:
            fresh_incumbent_score = incumbent_score
        accepted = candidate_score < fresh_incumbent_score
        incumbent[accepted] = candidate[accepted]
        incumbent_score = torch.where(accepted, candidate_score, fresh_incumbent_score)
        filled = _insert_racing_candidate(
            archive,
            archive_scores,
            candidate,
            candidate_score,
            filled,
        )

        if race_eot > 0 and (iteration + 1) % race_interval == 0:
            archive[-1] = incumbent
            archive_scores[-1] = incumbent_score
            filled = pool_size
            for slot in range(pool_size):
                archive_scores[slot] = score_draws_fn(
                    oracle.query(archive[slot], race_eot)
                )
            best_slots = archive_scores.argmin(dim=0)
            samples = torch.arange(len(x), device=x.device)
            incumbent = archive[best_slots, samples].detach().clone()
            incumbent_score = archive_scores[best_slots, samples].detach().clone()
            races_run += 1

    if races_run != planned_races:
        raise RuntimeError(
            f"SquareRacing executed {races_run} races, expected {planned_races}"
        )
    if oracle.api_queries_per_example != int(plan["search_queries"]):
        raise RuntimeError("SquareRacing search-query accounting mismatch")

    archive[-1] = incumbent
    archive_scores[-1] = incumbent_score
    confirmation_draws = [int(value) for value in plan["confirmation_draws"]]
    confirmed_scores = []
    for slot, draws in enumerate(confirmation_draws):
        confirmed_scores.append(score_draws_fn(oracle.query(archive[slot], draws)))
    score_matrix = torch.stack(confirmed_scores, dim=0)
    best_slots = score_matrix.argmin(dim=0)
    samples = torch.arange(len(x), device=x.device)
    confirmed = archive[best_slots, samples].detach().clone()

    if oracle.api_queries_per_example != int(api_budget):
        raise RuntimeError(
            "SquareRacing query accounting mismatch: "
            f"used {oracle.api_queries_per_example}, expected {api_budget}"
        )
    if oracle.logical_calls != int(plan["logical_calls"]):
        raise RuntimeError("SquareRacing logical-call accounting mismatch")

    details: dict[str, object] = {
        "search_steps": search_steps,
        "search_eot": int(search_eot),
        "refresh_eot": int(refresh_eot),
        "race_rounds": races_run,
        "race_interval": int(race_interval),
        "race_eot": int(race_eot),
        "race_queries": int(plan["race_queries"]),
        "confirmation_candidates": int(pool_size),
        "confirmation_queries": int(plan["confirmation_queries"]),
        "confirmation_draws_min": int(min(confirmation_draws)),
        "confirmation_draws_max": int(max(confirmation_draws)),
        "score_temperature": float(score_temperature),
        "score_objective": (
            "mean_per_draw_sigmoid_exact_set_margin"
            if stochastic_defense
            else "deterministic_exact_set_margin"
        ),
        "logical_calls": int(oracle.logical_calls),
        "api_queries": int(oracle.api_queries_per_example),
    }
    return confirmed, details
