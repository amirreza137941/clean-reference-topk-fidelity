#!/usr/bin/env python3
"""Audited SquareAdaptive/SquareRacing primitives used by CIFAR-100 pilot R4.

The proposal distribution and p schedule follow the reference L-infinity
Square Attack.  All returned defended-logit vectors are counted as API queries.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np
import torch


def stable_int(text: str, modulus: int = 2**31 - 1) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16) % modulus


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ranked_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    return logits.topk(int(k), dim=1, largest=True, sorted=True).indices


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


def averaged_actual(defense: Any, candidate: torch.Tensor, draws: int) -> torch.Tensor:
    draws = int(draws)
    if draws < 1:
        raise ValueError("draws must be positive")
    if hasattr(defense, "draw_logits"):
        return defense.draw_logits(candidate, draws).detach().mean(0)
    parts = [defense.actual(candidate).detach() for _ in range(draws)]
    return torch.stack(parts, dim=0).mean(0)


@dataclass
class VariableQueryOracle:
    """Oracle returning mean logits while charging every constituent draw."""

    defense: Any
    logical_calls: int = 0
    api_queries_per_example: int = 0

    def query(self, candidate: torch.Tensor, draws: int) -> torch.Tensor:
        draws = int(draws)
        if draws < 1:
            raise ValueError("Every oracle call must use at least one API draw")
        self.logical_calls += 1
        self.api_queries_per_example += draws
        return averaged_actual(self.defense, candidate, draws)


@dataclass
class PerDrawQueryOracle:
    """Oracle returning each draw separately while charging every draw."""

    defense: Any
    logical_calls: int = 0
    api_queries_per_example: int = 0

    def query(self, candidate: torch.Tensor, draws: int) -> torch.Tensor:
        draws = int(draws)
        if draws < 1:
            raise ValueError("Every oracle call must use at least one API draw")
        self.logical_calls += 1
        self.api_queries_per_example += draws
        if hasattr(self.defense, "draw_logits"):
            return self.defense.draw_logits(candidate, draws).detach()
        return torch.stack(
            [self.defense.actual(candidate).detach() for _ in range(draws)], dim=0
        )


def square_reference_p_selection(
    p_init: float, iteration: int, n_iterations: int
) -> float:
    """Piecewise-constant p schedule from the official ECCV 2020 code."""

    if int(n_iterations) < 1:
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
) -> Dict[str, object]:
    api_budget = int(api_budget)
    search_eot = int(search_eot)
    refresh_eot = int(refresh_eot)
    pool_size = int(confirm_candidates)
    if min(api_budget, search_eot, refresh_eot, pool_size) < 1:
        raise ValueError("Budget, EOT counts, and confirmation pool must be positive")
    if not 0 < float(confirm_fraction) < 1:
        raise ValueError("confirm_fraction must be in (0,1)")
    nominal_confirmation = (
        max(pool_size, int(api_budget * float(confirm_fraction)))
        if stochastic_defense
        else pool_size
    )
    if api_budget < search_eot + nominal_confirmation:
        raise ValueError("API budget cannot pay for initialization and confirmation")
    step_cost = search_eot + refresh_eot
    search_room = api_budget - nominal_confirmation - search_eot
    search_steps = max(search_room // step_cost, 0)
    search_queries = search_eot + search_steps * step_cost
    confirmation_queries = api_budget - search_queries
    if confirmation_queries < pool_size:
        raise RuntimeError("Internal plan left too few confirmation queries")
    draws_base, extra = divmod(confirmation_queries, pool_size)
    confirmation_draws = [
        draws_base + int(slot < extra) for slot in range(pool_size)
    ]
    logical_calls = 1 + 2 * search_steps + pool_size
    if search_queries + sum(confirmation_draws) != api_budget:
        raise RuntimeError("SquareAdaptive plan does not consume the exact budget")
    return {
        "search_steps": search_steps,
        "search_queries": search_queries,
        "confirmation_queries": confirmation_queries,
        "confirmation_draws": confirmation_draws,
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
) -> Dict[str, object]:
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
        raise ValueError("Stochastic SquareRacing needs positive refresh/race EOT")
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
    confirmation_draws = [
        draws_base + int(slot < extra) for slot in range(pool_size)
    ]
    race_queries = race_rounds * pool_size * race_eot
    logical_calls = (
        1
        + search_steps
        + search_steps * int(refresh_eot > 0)
        + race_rounds * pool_size
        + pool_size
    )
    if search_queries + sum(confirmation_draws) != api_budget:
        raise RuntimeError("SquareRacing plan does not consume the exact budget")
    return {
        "search_steps": search_steps,
        "search_queries": search_queries,
        "race_rounds": race_rounds,
        "race_queries": race_queries,
        "confirmation_queries": confirmation_queries,
        "confirmation_draws": confirmation_draws,
        "logical_calls": logical_calls,
    }


def square_vertical_stripe_initialization(
    x: torch.Tensor, eps: float, rng: np.random.Generator
) -> torch.Tensor:
    batch, channels, _, width = x.shape
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(batch, channels, 1, width),
    )
    delta = torch.from_numpy(signs).to(device=x.device, dtype=x.dtype) * float(eps)
    return (x + delta).clamp(0.0, 1.0)


def square_reference_linf_proposal(
    x: torch.Tensor,
    incumbent: torch.Tensor,
    eps: float,
    p: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if height != width:
        raise ValueError("Square proposal expects square images")
    side = int(round(math.sqrt(float(p) * height * width)))
    side = min(max(side, 1), height - 1)
    candidate = incumbent.detach().clone()
    for sample in range(batch):
        row = int(rng.integers(0, height - side))
        col = int(rng.integers(0, width - side))
        clean_window = x[sample, :, row : row + side, col : col + side]
        old_window = incumbent[sample, :, row : row + side, col : col + side]
        signs_np = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=(16, channels, 1, 1),
        )
        signs = torch.from_numpy(signs_np).to(device=x.device, dtype=x.dtype)
        windows = (clean_window.unsqueeze(0) + float(eps) * signs).clamp(0.0, 1.0)
        changed = (windows - old_window.unsqueeze(0)).abs().gt(1e-7).flatten(1).any(1)
        first_changed = changed.to(torch.int64).argmax()
        fallback = (clean_window - float(eps) * signs[0]).clamp(0.0, 1.0)
        new_window = torch.where(changed.any(), windows[first_changed], fallback)
        candidate[sample, :, row : row + side, col : col + side] = new_window
    return candidate


def _archive_accepted_candidates(
    archive: torch.Tensor,
    archive_scores: torch.Tensor,
    candidate: torch.Tensor,
    candidate_score: torch.Tensor,
    accepted: torch.Tensor,
) -> None:
    worst_slots = archive_scores.argmax(0)
    samples = torch.arange(len(candidate), device=candidate.device)
    worst_scores = archive_scores[worst_slots, samples]
    replace = accepted & (candidate_score < worst_scores)
    replace_samples = samples[replace]
    replace_slots = worst_slots[replace]
    archive[replace_slots, replace_samples] = candidate[replace]
    archive_scores[replace_slots, replace_samples] = candidate_score[replace]


def square_adaptive_attack_budget(
    oracle: VariableQueryOracle,
    x: torch.Tensor,
    reference_topk: torch.Tensor,
    eps: float,
    api_budget: int,
    p_init: float,
    config: Mapping[str, Any],
    stochastic_defense: bool,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    search_eot = int(config["search_eot"])
    refresh_eot = int(config["refresh_eot"])
    pool_size = int(config["confirm_candidates"])
    plan = square_adaptive_query_plan(
        api_budget,
        search_eot,
        refresh_eot,
        pool_size,
        float(config["confirm_fraction"]),
        stochastic_defense,
    )

    def score(logits: torch.Tensor) -> torch.Tensor:
        return exact_set_attack_margin(logits, reference_topk)

    search_steps = int(plan["search_steps"])
    incumbent = square_vertical_stripe_initialization(x, eps, rng)
    incumbent_score = score(oracle.query(incumbent, search_eot))
    archive = incumbent.unsqueeze(0).repeat(pool_size, *([1] * incumbent.ndim))
    archive_scores = torch.full(
        (pool_size, len(x)), float("inf"), device=x.device, dtype=incumbent_score.dtype
    )
    archive_scores[0] = incumbent_score
    for iteration in range(search_steps):
        p = square_reference_p_selection(p_init, iteration, max(search_steps, 1))
        candidate = square_reference_linf_proposal(x, incumbent, eps, p, rng)
        candidate_score = score(oracle.query(candidate, search_eot))
        fresh_incumbent_score = score(oracle.query(incumbent, refresh_eot))
        accepted = candidate_score < fresh_incumbent_score
        incumbent[accepted] = candidate[accepted]
        incumbent_score = torch.where(accepted, candidate_score, fresh_incumbent_score)
        _archive_accepted_candidates(
            archive, archive_scores, candidate, candidate_score, accepted
        )
    archive[-1] = incumbent
    archive_scores[-1] = incumbent_score
    if oracle.api_queries_per_example != int(plan["search_queries"]):
        raise RuntimeError("SquareAdaptive search-query accounting mismatch")
    confirmed_scores = []
    for slot, draws in enumerate(plan["confirmation_draws"]):
        confirmed_scores.append(score(oracle.query(archive[slot], int(draws))))
    score_matrix = torch.stack(confirmed_scores, dim=0)
    best_slots = score_matrix.argmin(0)
    samples = torch.arange(len(x), device=x.device)
    confirmed = archive[best_slots, samples].detach().clone()
    if oracle.api_queries_per_example != int(api_budget):
        raise RuntimeError("SquareAdaptive did not consume the exact API budget")
    details: Dict[str, object] = {
        "search_steps": search_steps,
        "search_eot": search_eot,
        "refresh_eot": refresh_eot,
        "confirmation_candidates": pool_size,
        "confirmation_queries": int(plan["confirmation_queries"]),
        "confirmation_draws_min": int(min(plan["confirmation_draws"])),
        "confirmation_draws_max": int(max(plan["confirmation_draws"])),
        "logical_calls": int(oracle.logical_calls),
        "api_queries": int(oracle.api_queries_per_example),
        "score_objective": str(config["objective"]),
    }
    return confirmed, details


def _insert_racing_candidate(
    archive: torch.Tensor,
    archive_scores: torch.Tensor,
    candidate: torch.Tensor,
    candidate_score: torch.Tensor,
    filled: int,
) -> int:
    pool_size = int(archive.shape[0])
    if filled < pool_size:
        archive[filled] = candidate
        archive_scores[filled] = candidate_score
        return filled + 1
    worst_slots = archive_scores.argmax(0)
    samples = torch.arange(len(candidate), device=candidate.device)
    worst_scores = archive_scores[worst_slots, samples]
    replace = candidate_score < worst_scores
    replace_samples = samples[replace]
    replace_slots = worst_slots[replace]
    archive[replace_slots, replace_samples] = candidate[replace]
    archive_scores[replace_slots, replace_samples] = candidate_score[replace]
    return filled


def square_racing_attack_budget(
    oracle: PerDrawQueryOracle,
    x: torch.Tensor,
    reference_topk: torch.Tensor,
    eps: float,
    api_budget: int,
    p_init: float,
    config: Mapping[str, Any],
    stochastic_defense: bool,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if stochastic_defense:
        search_eot = int(config["search_eot"])
        refresh_eot = int(config["refresh_eot"])
        pool_size = int(config["pool_size"])
        race_interval = int(config["race_interval"])
        race_eot = int(config["race_eot"])
    else:
        search_eot = 1
        refresh_eot = 0
        pool_size = 1
        race_interval = int(api_budget) + 1
        race_eot = 0
    plan = square_racing_query_plan(
        api_budget,
        search_eot,
        refresh_eot,
        pool_size,
        race_interval,
        race_eot,
        float(config["confirm_fraction"]),
        stochastic_defense,
    )

    def score(draw_logits: torch.Tensor) -> torch.Tensor:
        if stochastic_defense:
            return per_draw_exact_set_survival_score(
                draw_logits, reference_topk, float(config["score_temperature"])
            )
        return exact_set_attack_margin(draw_logits[0], reference_topk)

    search_steps = int(plan["search_steps"])
    incumbent = square_vertical_stripe_initialization(x, eps, rng)
    incumbent_score = score(oracle.query(incumbent, search_eot))
    archive = incumbent.unsqueeze(0).repeat(pool_size, *([1] * incumbent.ndim))
    archive_scores = incumbent_score.unsqueeze(0).repeat(pool_size, 1)
    filled = 1
    races_run = 0
    for iteration in range(search_steps):
        p = square_reference_p_selection(p_init, iteration, max(search_steps, 1))
        candidate = square_reference_linf_proposal(x, incumbent, eps, p, rng)
        candidate_score = score(oracle.query(candidate, search_eot))
        if refresh_eot > 0:
            fresh_incumbent_score = score(oracle.query(incumbent, refresh_eot))
        else:
            fresh_incumbent_score = incumbent_score
        accepted = candidate_score < fresh_incumbent_score
        incumbent[accepted] = candidate[accepted]
        incumbent_score = torch.where(accepted, candidate_score, fresh_incumbent_score)
        filled = _insert_racing_candidate(
            archive, archive_scores, candidate, candidate_score, filled
        )
        if race_eot > 0 and (iteration + 1) % race_interval == 0:
            archive[-1] = incumbent
            archive_scores[-1] = incumbent_score
            filled = pool_size
            for slot in range(pool_size):
                archive_scores[slot] = score(oracle.query(archive[slot], race_eot))
            best_slots = archive_scores.argmin(0)
            samples = torch.arange(len(x), device=x.device)
            incumbent = archive[best_slots, samples].detach().clone()
            incumbent_score = archive_scores[best_slots, samples].detach().clone()
            races_run += 1
    if races_run != int(plan["race_rounds"]):
        raise RuntimeError("SquareRacing race-round accounting mismatch")
    if oracle.api_queries_per_example != int(plan["search_queries"]):
        raise RuntimeError("SquareRacing search-query accounting mismatch")
    archive[-1] = incumbent
    archive_scores[-1] = incumbent_score
    confirmed_scores = []
    for slot, draws in enumerate(plan["confirmation_draws"]):
        confirmed_scores.append(score(oracle.query(archive[slot], int(draws))))
    score_matrix = torch.stack(confirmed_scores, dim=0)
    best_slots = score_matrix.argmin(0)
    samples = torch.arange(len(x), device=x.device)
    confirmed = archive[best_slots, samples].detach().clone()
    if oracle.api_queries_per_example != int(api_budget):
        raise RuntimeError("SquareRacing did not consume the exact API budget")
    if oracle.logical_calls != int(plan["logical_calls"]):
        raise RuntimeError("SquareRacing logical-call accounting mismatch")
    details: Dict[str, object] = {
        "search_steps": search_steps,
        "search_eot": search_eot,
        "refresh_eot": refresh_eot,
        "race_rounds": races_run,
        "race_interval": race_interval,
        "race_eot": race_eot,
        "race_queries": int(plan["race_queries"]),
        "confirmation_candidates": pool_size,
        "confirmation_queries": int(plan["confirmation_queries"]),
        "confirmation_draws_min": int(min(plan["confirmation_draws"])),
        "confirmation_draws_max": int(max(plan["confirmation_draws"])),
        "logical_calls": int(oracle.logical_calls),
        "api_queries": int(oracle.api_queries_per_example),
        "score_temperature": float(config["score_temperature"]),
        "score_objective": (
            str(config["objective"])
            if stochastic_defense
            else "deterministic_exact_set_margin"
        ),
    }
    return confirmed, details


def run_square_batch(
    *,
    attack: str,
    defense: Any,
    clean: torch.Tensor,
    reference_topk: torch.Tensor,
    eps: float,
    api_budget: int,
    p_init: float,
    adaptive_config: Mapping[str, Any],
    racing_config: Mapping[str, Any],
    proposal_seed: int,
    oracle_seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    rng = np.random.default_rng(int(proposal_seed))
    seed_everything(int(oracle_seed))
    if attack == "SquareAdaptive":
        return square_adaptive_attack_budget(
            VariableQueryOracle(defense),
            clean,
            reference_topk,
            eps,
            api_budget,
            p_init,
            adaptive_config,
            bool(defense.sigma > 0),
            rng,
        )
    if attack == "SquareRacing":
        return square_racing_attack_budget(
            PerDrawQueryOracle(defense),
            clean,
            reference_topk,
            eps,
            api_budget,
            p_init,
            racing_config,
            bool(defense.sigma > 0),
            rng,
        )
    raise ValueError(f"Unsupported attack: {attack}")
