"""Original integer query allocations and the Square patch-size schedule.

Plans count each returned logit vector per image, including initialization,
refresh, racing and in-attack confirmation. Independent endpoint evaluation
is outside this budget. Zero refresh is supported for deterministic CIFAR-10.
"""

from __future__ import annotations


def square_reference_p_selection(
    p_init: float, iteration: int, n_iterations: int
) -> float:
    """Published piecewise-constant Square Attack schedule.

    This is the schedule in ``p_selection`` from the authors' official ECCV
    2020 repository.  Iteration is rescaled to a 10,000-step reference run.
    """

    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    scaled = int(int(iteration) / int(n_iterations) * 10000)
    if 10 < scaled <= 50:
        return p_init / 2
    if 50 < scaled <= 200:
        return p_init / 4
    if 200 < scaled <= 500:
        return p_init / 8
    if 500 < scaled <= 1000:
        return p_init / 16
    if 1000 < scaled <= 2000:
        return p_init / 32
    if 2000 < scaled <= 4000:
        return p_init / 64
    if 4000 < scaled <= 6000:
        return p_init / 128
    if 6000 < scaled <= 8000:
        return p_init / 256
    if 8000 < scaled <= 10000:
        return p_init / 512
    return p_init


def square_adaptive_query_plan(
    api_budget: int,
    search_eot: int,
    refresh_eot: int,
    confirm_candidates: int,
    confirm_fraction: float,
    stochastic_defense: bool,
) -> dict[str, object]:
    """Construct an exact, integer API-query plan without touching a model."""

    api_budget = int(api_budget)
    search_eot = int(search_eot)
    refresh_eot = int(refresh_eot)
    pool_size = int(confirm_candidates)
    if min(api_budget, search_eot, pool_size) < 1 or refresh_eot < 0:
        raise ValueError(
            "Budget, search EOT, and confirmation pool must be positive; "
            "refresh EOT must be non-negative"
        )
    if stochastic_defense and refresh_eot < 1:
        raise ValueError("A stochastic defense requires positive refresh EOT")
    if not 0 < float(confirm_fraction) < 1:
        raise ValueError("Confirmation fraction must be in (0,1)")
    nominal_confirmation = (
        max(pool_size, int(api_budget * float(confirm_fraction)))
        if stochastic_defense
        else pool_size
    )
    if api_budget < search_eot + nominal_confirmation:
        raise ValueError(
            f"API budget {api_budget} cannot pay for search initialization and "
            f"{pool_size} confirmation candidates"
        )
    step_cost = search_eot + refresh_eot
    search_room = api_budget - nominal_confirmation - search_eot
    search_steps = max(search_room // step_cost, 0)
    search_queries = search_eot + search_steps * step_cost
    confirmation_queries = api_budget - search_queries
    if confirmation_queries < pool_size:
        raise RuntimeError("Internal budget error: less than one draw per archive slot")
    draws_base, extra = divmod(confirmation_queries, pool_size)
    draw_allocation = [draws_base + int(slot < extra) for slot in range(pool_size)]
    logical_calls = 1 + search_steps + search_steps * int(refresh_eot > 0) + pool_size
    if search_queries + sum(draw_allocation) != api_budget:
        raise RuntimeError("Adaptive query-plan accounting mismatch")
    return {
        "search_steps": search_steps,
        "search_queries": search_queries,
        "confirmation_queries": confirmation_queries,
        "confirmation_draws": draw_allocation,
        "logical_calls": logical_calls,
    }


def square_racing_query_plan(
    api_budget: int,
    search_eot: int,
    refresh_eot: int,
    pool_size: int,
    race_interval: int,
    race_eot: int,
    confirm_fraction: float,
    stochastic_defense: bool,
) -> dict[str, object]:
    """Build an exact API plan for search, archive races, and confirmation."""

    api_budget = int(api_budget)
    search_eot = int(search_eot)
    refresh_eot = int(refresh_eot)
    pool_size = int(pool_size)
    race_interval = int(race_interval)
    race_eot = int(race_eot)
    if min(api_budget, search_eot, pool_size, race_interval) < 1:
        raise ValueError("Budget, search EOT, pool size, and interval must be positive")
    if refresh_eot < 0 or race_eot < 0:
        raise ValueError("Refresh and race EOT must be non-negative")
    if stochastic_defense and min(refresh_eot, race_eot) < 1:
        raise ValueError("Stochastic SquareRacing requires positive refresh/race EOT")
    if not 0 < float(confirm_fraction) < 1:
        raise ValueError("confirm_fraction must be in (0,1)")

    nominal_confirmation = (
        max(pool_size, int(api_budget * float(confirm_fraction)))
        if stochastic_defense
        else pool_size
    )
    if api_budget < search_eot + nominal_confirmation:
        raise ValueError("API budget cannot pay for initialization and confirmation")

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
        if race_due:
            race_rounds += 1

    confirmation_queries = api_budget - search_queries
    if confirmation_queries < pool_size:
        raise RuntimeError("Internal plan left too few confirmation queries")
    draws_base, extra = divmod(confirmation_queries, pool_size)
    confirmation_draws = [draws_base + int(slot < extra) for slot in range(pool_size)]
    race_queries = race_rounds * pool_size * race_eot
    logical_calls = (
        1
        + search_steps
        + search_steps * int(refresh_eot > 0)
        + race_rounds * pool_size
        + pool_size
    )
    if search_queries + sum(confirmation_draws) != api_budget:
        raise RuntimeError("SquareRacing plan does not consume the exact API budget")
    return {
        "search_steps": search_steps,
        "search_queries": search_queries,
        "race_rounds": race_rounds,
        "race_queries": race_queries,
        "confirmation_queries": confirmation_queries,
        "confirmation_draws": confirmation_draws,
        "logical_calls": logical_calls,
    }
