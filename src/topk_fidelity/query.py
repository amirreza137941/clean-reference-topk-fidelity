"""Query accounting for a fixed image batch in one attack family and seed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import positive_integer


@dataclass
class QueryLedger:
    """Count full-logit responses PER IMAGE, with an optional hard limit.

    A batch query with d draws costs d responses for EACH member. Use a fresh
    ledger for each image batch, attack family and seed. Independent endpoint
    MC evaluation stays outside this ledger.
    """

    logical_calls: int = 0
    api_queries_per_example: int = 0
    limit: int | None = None

    def __post_init__(self) -> None:
        positive_integer(self.logical_calls, "logical_calls", minimum=0)
        positive_integer(
            self.api_queries_per_example, "api_queries_per_example", minimum=0
        )
        if self.limit is not None:
            positive_integer(self.limit, "limit")
            if self.api_queries_per_example > self.limit:
                raise ValueError("initial query count exceeds the limit")

    def charge(self, draws: int) -> None:
        positive_integer(draws, "draws")
        if self.limit is not None and self.api_queries_per_example + draws > self.limit:
            raise RuntimeError("oracle call would exceed the per-image API budget")
        # Charge before invoking the defense, as in the original oracle.
        self.logical_calls += 1
        self.api_queries_per_example += draws


def collect_draw_logits(
    defense: Any, candidate: torch.Tensor, draws: int
) -> torch.Tensor:
    """Collect full responses from a callable or an original DefenseSpec."""
    positive_integer(draws, "draws")
    if hasattr(defense, "draw_logits"):
        result = defense.draw_logits(candidate, draws)
    else:
        actual = getattr(defense, "actual", defense)
        result = torch.stack([actual(candidate) for _ in range(draws)], dim=0)
    if (
        result.ndim != 3
        or result.shape[0] != draws
        or result.shape[1] != candidate.shape[0]
    ):
        raise ValueError("defense draws must have shape [draws,batch,classes]")
    return result.detach()


@dataclass
class PerDrawQueryOracle:
    """Return each full response separately, charging every constituent draw."""

    defense: Any
    ledger: QueryLedger | None = None

    def __post_init__(self) -> None:
        if self.ledger is None:
            self.ledger = QueryLedger()

    @property
    def logical_calls(self) -> int:
        assert self.ledger is not None
        return self.ledger.logical_calls

    @property
    def api_queries_per_example(self) -> int:
        assert self.ledger is not None
        return self.ledger.api_queries_per_example

    def query(self, candidate: torch.Tensor, draws: int) -> torch.Tensor:
        assert self.ledger is not None
        self.ledger.charge(draws)
        return collect_draw_logits(self.defense, candidate, draws)


@dataclass
class AveragedQueryOracle(PerDrawQueryOracle):
    """Mean logits for SquareAdaptive; metrics still use individual draws."""

    reduction: str = "stack_mean"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reduction not in {"stack_mean", "sequential_sum"}:
            raise ValueError("unsupported mean-logit reduction")

    def query(self, candidate: torch.Tensor, draws: int) -> torch.Tensor:
        if self.reduction == "stack_mean":
            return super().query(candidate, draws).mean(0)
        # CIFAR-10's original averaged_actual uses left-to-right summation.
        # Keep its floating-point accumulation order for attack confirmation.
        assert self.ledger is not None
        self.ledger.charge(draws)
        actual = getattr(self.defense, "actual", self.defense)
        total = None
        for _ in range(draws):
            logits = actual(candidate).detach()
            if logits.ndim != 2 or logits.shape[0] != candidate.shape[0]:
                raise ValueError("defense response must have shape [batch,classes]")
            total = logits if total is None else total + logits
        assert total is not None
        return total / float(draws)
