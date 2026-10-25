"""Model-independent entry point for the original two Square searches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ..query import AveragedQueryOracle, PerDrawQueryOracle, QueryLedger
from ..randomness import seed_everything
from .adaptive import square_adaptive_attack_budget
from .objectives import exact_set_attack_margin, per_draw_exact_set_survival_score
from .racing import square_racing_attack_budget


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
    mean_logit_reduction: str = "stack_mean",
) -> tuple[torch.Tensor, dict[str, object]]:
    """Search for one candidate per image using the supplied clean target set.

    The target remains fixed for attack search; the defense recomputes its own
    online reference for every request. A fresh ledger counts returned full
    logits, not internal reference/anchor forwards. This returns a confirmed
    candidate, without using independent endpoint MC evaluation for selection.

    Adaptive parameters are used exactly as supplied. ExperimentConfig makes
    the CIFAR-10/CIFAR-100 deterministic-refresh difference explicit.
    """
    if clean.ndim != 4 or clean.shape[-1] != clean.shape[-2] or clean.shape[-1] < 2:
        raise ValueError("Square input must have shape [batch,channels,H,H], H >= 2")
    if len(clean) == 0:
        raise ValueError("attack batch cannot be empty")
    if reference_topk.ndim != 2 or len(reference_topk) != len(clean):
        raise ValueError("reference_topk must have shape [batch,k]")
    if attack not in ("SquareAdaptive", "SquareRacing"):
        raise ValueError(f"unsupported attack: {attack}")
    rng = np.random.default_rng(int(proposal_seed))
    seed_everything(int(oracle_seed))
    sigma = getattr(defense, "sigma", None)
    if sigma is None and hasattr(defense, "config"):
        sigma = defense.config.sigma
    if sigma is None:
        raise ValueError("defense must declare sigma (zero for a deterministic oracle)")
    stochastic = bool(sigma > 0)
    ledger = QueryLedger(limit=api_budget)

    if attack == "SquareAdaptive":
        oracle = AveragedQueryOracle(defense, ledger, reduction=mean_logit_reduction)
        candidate, details = square_adaptive_attack_budget(
            oracle=oracle,
            x=clean,
            score_fn=lambda logits: exact_set_attack_margin(logits, reference_topk),
            eps=eps,
            api_budget=api_budget,
            p_init=p_init,
            search_eot=int(adaptive_config["search_eot"]),
            refresh_eot=int(adaptive_config["refresh_eot"]),
            confirm_candidates=int(adaptive_config["confirm_candidates"]),
            confirm_fraction=float(adaptive_config["confirm_fraction"]),
            stochastic_defense=stochastic,
            rng=rng,
        )
        details["score_objective"] = str(adaptive_config["objective"])
        return candidate, details

    config = racing_config
    candidate, details = square_racing_attack_budget(
        oracle=PerDrawQueryOracle(defense, ledger),
        x=clean,
        score_draws_fn=(
            lambda logits: (
                per_draw_exact_set_survival_score(
                    logits, reference_topk, float(config["score_temperature"])
                )
                if stochastic
                else exact_set_attack_margin(logits[0], reference_topk)
            )
        ),
        eps=eps,
        api_budget=api_budget,
        p_init=p_init,
        search_eot=int(config["search_eot"]) if stochastic else 1,
        refresh_eot=int(config["refresh_eot"]) if stochastic else 0,
        pool_size=int(config["pool_size"]) if stochastic else 1,
        race_interval=int(config["race_interval"]) if stochastic else api_budget + 1,
        race_eot=int(config["race_eot"]) if stochastic else 0,
        confirm_fraction=float(config["confirm_fraction"]),
        stochastic_defense=stochastic,
        score_temperature=float(config["score_temperature"]),
        rng=rng,
    )
    details["score_objective"] = (
        str(config["objective"]) if stochastic else "deterministic_exact_set_margin"
    )
    return candidate, details
