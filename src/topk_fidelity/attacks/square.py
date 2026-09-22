"""Query plans and objectives used by the audited Square-attack pipeline."""

from __future__ import annotations

from typing import Dict

import torch


def exact_set_attack_margin(
    logits: torch.Tensor, reference_topk: torch.Tensor
) -> torch.Tensor:
    """Positive iff every reference item outranks every outside item."""

    if logits.ndim != 2 or reference_topk.ndim != 2:
        raise ValueError("logits and reference_topk must both be rank two")
    if logits.shape[0] != reference_topk.shape[0]:
        raise ValueError("batch dimensions must agree")
    selected = logits.gather(1, reference_topk)
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference_topk, False)
    strongest_outside = logits.masked_fill(~outside, float("-inf")).max(1).values
    return selected.min(1).values - strongest_outside


def per_draw_exact_set_survival_score(
    draw_logits: torch.Tensor,
    reference_topk: torch.Tensor,
    *,
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


def square_reference_p_selection(
    p_init: float, iteration: int, n_iterations: int
) -> float:
    """Piecewise p schedule from the reference L-infinity Square Attack."""

    if int(n_iterations) < 1:
        raise ValueError("n_iterations must be positive")
    scaled = int(int(iteration) / int(n_iterations) * 10000)
    schedule = (
        (10, 50, 2),
        (50, 200, 4),
        (200, 500, 8),
        (500, 1000, 16),
        (1000, 2000, 32),
        (2000, 4000, 64),
        (4000, 6000, 128),
        (6000, 8000, 256),
        (8000, 10000, 512),
    )
    for lower, upper, divisor in schedule:
        if lower < scaled <= upper:
            return float(p_init) / divisor
    return float(p_init)


def square_adaptive_query_plan(
    *,
    api_budget: int,
    search_eot: int,
    refresh_eot: int,
    confirm_candidates: int,
    confirm_fraction: float,
    stochastic_defense: bool,
) -> Dict[str, object]:
    """Allocate an exact API budget between search and confirmation."""

    api_budget = int(api_budget)
    search_eot = int(search_eot)
    refresh_eot = int(refresh_eot)
    pool_size = int(confirm_candidates)
    if min(api_budget, search_eot, refresh_eot, pool_size) < 1:
        raise ValueError("budget, EOT counts, and pool size must be positive")
    if not 0 < float(confirm_fraction) < 1:
        raise ValueError("confirm_fraction must be in (0,1)")
    nominal_confirmation = (
        max(pool_size, int(api_budget * float(confirm_fraction)))
        if stochastic_defense
        else pool_size
    )
    if api_budget < search_eot + nominal_confirmation:
        raise ValueError("budget cannot pay for initialization and confirmation")
    step_cost = search_eot + refresh_eot
    search_steps = max(
        (api_budget - nominal_confirmation - search_eot) // step_cost, 0
    )
    search_queries = search_eot + search_steps * step_cost
    confirmation_queries = api_budget - search_queries
    draws_base, extra = divmod(confirmation_queries, pool_size)
    confirmation_draws = [
        draws_base + int(slot < extra) for slot in range(pool_size)
    ]
    if search_queries + sum(confirmation_draws) != api_budget:
        raise RuntimeError("query plan does not consume the exact budget")
    return {
        "search_steps": search_steps,
        "search_queries": search_queries,
        "confirmation_queries": confirmation_queries,
        "confirmation_draws": confirmation_draws,
        "logical_calls": 1 + 2 * search_steps + pool_size,
    }


def square_racing_query_plan(
    *,
    api_budget: int,
    search_eot: int,
    refresh_eot: int,
    pool_size: int,
    race_interval: int,
    race_eot: int,
    confirm_fraction: float,
    stochastic_defense: bool,
) -> Dict[str, object]:
    """Allocate an exact API budget for the racing variant."""

    api_budget = int(api_budget)
    search_eot = int(search_eot)
    refresh_eot = int(refresh_eot)
    pool_size = int(pool_size)
    race_interval = int(race_interval)
    race_eot = int(race_eot)
    if min(api_budget, search_eot, pool_size, race_interval) < 1:
        raise ValueError("budget, search EOT, pool size, and interval must be positive")
    if refresh_eot < 0 or race_eot < 0:
        raise ValueError("refresh and race EOT must be non-negative")
    if stochastic_defense and min(refresh_eot, race_eot) < 1:
        raise ValueError("stochastic racing needs positive refresh and race EOT")
    if not 0 < float(confirm_fraction) < 1:
        raise ValueError("confirm_fraction must be in (0,1)")
    nominal_confirmation = (
        max(pool_size, int(api_budget * float(confirm_fraction)))
        if stochastic_defense
        else pool_size
    )
    if api_budget < search_eot + nominal_confirmation:
        raise ValueError("budget cannot pay for initialization and confirmation")

    search_queries = search_eot
    search_steps = 0
    race_rounds = 0
    while True:
        next_step = search_steps + 1
        race_due = race_eot > 0 and next_step % race_interval == 0
        incremental = search_eot + refresh_eot
        if race_due:
            incremental += pool_size * race_eot
        if search_queries + incremental + nominal_confirmation > api_budget:
            break
        search_queries += incremental
        search_steps = next_step
        race_rounds += int(race_due)

    confirmation_queries = api_budget - search_queries
    draws_base, extra = divmod(confirmation_queries, pool_size)
    confirmation_draws = [
        draws_base + int(slot < extra) for slot in range(pool_size)
    ]
    if search_queries + sum(confirmation_draws) != api_budget:
        raise RuntimeError("query plan does not consume the exact budget")
    return {
        "search_steps": search_steps,
        "search_queries": search_queries,
        "race_rounds": race_rounds,
        "race_queries": race_rounds * pool_size * race_eot,
        "confirmation_queries": confirmation_queries,
        "confirmation_draws": confirmation_draws,
        "logical_calls": (
            1
            + search_steps
            + search_steps * int(refresh_eot > 0)
            + race_rounds * pool_size
            + pool_size
        ),
    }
