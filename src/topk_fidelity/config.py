"""Validated pixel-space parameters and explicit experiment settings.

DefenseConfig.epsilon is the INTERNAL strengthening radius. Attack epsilon is
a separate field in ExperimentConfig. Neither is measured in normalized units.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


def positive_integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class DefenseConfig:
    """Parameters of one response, with radii and noise in [0,1] pixel units."""

    top_k: int
    alpha: float
    epsilon: float
    steps: int
    sigma: float
    top1_weight: float = 1.0
    set_weight: float = 4.0

    def __post_init__(self) -> None:
        positive_integer(self.top_k, "top_k")
        positive_integer(self.steps, "steps", minimum=0)
        for name in ("alpha", "epsilon", "sigma", "top1_weight", "set_weight"):
            finite_nonnegative(getattr(self, name), name)
        if self.top1_weight + self.set_weight <= 0:
            raise ValueError("at least one objective weight must be positive")

    def for_method(self, method: str) -> DefenseConfig:
        """Apply the original Robust / strength-only / noise-only ablations."""
        if method not in ("robust", "full_sigma0", "noise_only", "full"):
            raise ValueError(f"unsupported method: {method}")
        strength = method in ("full_sigma0", "full")
        noise = method in ("noise_only", "full")
        return replace(
            self,
            steps=self.steps if strength else 0,
            alpha=self.alpha if strength else 0.0,
            epsilon=self.epsilon if strength else 0.0,
            sigma=self.sigma if noise else 0.0,
        )


@dataclass(frozen=True)
class AdaptiveConfig:
    """SquareAdaptive search, refresh and in-attack confirmation allocation."""

    search_eot: int = 1
    refresh_eot: int = 1
    confirm_candidates: int = 5
    confirm_fraction: float = 0.20
    objective: str = "exact_set_margin_of_mean_logits"

    def __post_init__(self) -> None:
        positive_integer(self.search_eot, "search_eot")
        positive_integer(self.refresh_eot, "refresh_eot", minimum=0)
        positive_integer(self.confirm_candidates, "confirm_candidates")
        if (
            not math.isfinite(self.confirm_fraction)
            or not 0 < self.confirm_fraction < 1
        ):
            raise ValueError("confirm_fraction must be finite and in (0,1)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RacingConfig:
    """SquareRacing per-draw objective, archive racing and confirmation."""

    search_eot: int = 5
    refresh_eot: int = 5
    pool_size: int = 8
    race_interval: int = 25
    race_eot: int = 5
    confirm_fraction: float = 0.25
    score_temperature: float = 0.10
    objective: str = "mean_per_draw_sigmoid_exact_set_margin"

    def __post_init__(self) -> None:
        for name in ("search_eot", "pool_size", "race_interval"):
            positive_integer(getattr(self, name), name)
        for name in ("refresh_eot", "race_eot"):
            positive_integer(getattr(self, name), name, minimum=0)
        if (
            not math.isfinite(self.confirm_fraction)
            or not 0 < self.confirm_fraction < 1
        ):
            raise ValueError("confirm_fraction must be finite and in (0,1)")
        if not math.isfinite(self.score_temperature) or self.score_temperature <= 0:
            raise ValueError("score_temperature must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Model-independent protocol settings loaded from a small JSON file."""

    dataset: str
    reference_family: str
    defense: DefenseConfig
    attack_epsilon: float
    api_budget: int
    p_init: float
    search_seeds: tuple[int, ...]
    reevaluation_draws: int
    adaptive: AdaptiveConfig
    racing: RacingConfig
    deterministic_adaptive_refresh: int
    method_attack: dict[str, str]
    family_selection: str
    mean_logit_reduction: str

    def __post_init__(self) -> None:
        finite_nonnegative(self.attack_epsilon, "attack_epsilon")
        positive_integer(self.api_budget, "api_budget")
        positive_integer(self.reevaluation_draws, "reevaluation_draws")
        positive_integer(
            self.deterministic_adaptive_refresh,
            "deterministic_adaptive_refresh",
            minimum=0,
        )
        if not math.isfinite(self.p_init) or not 0 < self.p_init <= 1:
            raise ValueError("p_init must be finite and in (0,1]")
        if not self.search_seeds or len(set(self.search_seeds)) != len(
            self.search_seeds
        ):
            raise ValueError("search_seeds must be nonempty and unique")
        for seed in self.search_seeds:
            positive_integer(seed, "search seed", minimum=0)
        allowed_methods = {"robust", "full_sigma0", "noise_only", "full"}
        if not set(self.method_attack) <= allowed_methods:
            raise ValueError("method_attack contains an unsupported method")
        if not set(self.method_attack.values()) <= {"SquareAdaptive", "SquareRacing"}:
            raise ValueError("method_attack contains an unsupported attack")
        if self.dataset not in {"cifar10", "cifar100", "imagenet"}:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        if self.mean_logit_reduction not in {"stack_mean", "sequential_sum"}:
            raise ValueError("unsupported mean_logit_reduction")

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Reject misspelled keys instead of silently using unintended defaults."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("experiment JSON must be an object")
        payload["defense"] = DefenseConfig(**payload["defense"])
        payload["adaptive"] = AdaptiveConfig(**payload["adaptive"])
        payload["racing"] = RacingConfig(**payload["racing"])
        payload["search_seeds"] = tuple(payload["search_seeds"])
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def adaptive_for(self, stochastic: bool) -> AdaptiveConfig:
        if stochastic:
            return self.adaptive
        return replace(
            self.adaptive,
            search_eot=1,
            refresh_eot=self.deterministic_adaptive_refresh,
        )
