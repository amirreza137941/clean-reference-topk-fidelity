"""Auditable query accounting for stochastic defenses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QueryLedger:
    """Counts logical oracle calls and constituent defense draws per example."""

    logical_calls: int = 0
    api_queries_per_example: int = 0

    def charge(self, draws: int) -> None:
        if int(draws) < 1:
            raise ValueError("every oracle call must use at least one draw")
        self.logical_calls += 1
        self.api_queries_per_example += int(draws)


def _draw_logits(defense: Any, candidate: torch.Tensor, draws: int) -> torch.Tensor:
    if hasattr(defense, "draw_logits"):
        result = defense.draw_logits(candidate, int(draws))
    else:
        result = torch.stack(
            [defense(candidate) for _ in range(int(draws))], dim=0
        )
    if result.ndim != 3 or result.shape[0] != int(draws):
        raise ValueError("defense draws must have shape [draws,batch,classes]")
    return result.detach()


@dataclass
class AveragedQueryOracle:
    """Return mean logits while charging every stochastic response."""

    defense: Any
    ledger: QueryLedger | None = None

    def __post_init__(self) -> None:
        if self.ledger is None:
            self.ledger = QueryLedger()

    def query(self, candidate: torch.Tensor, draws: int) -> torch.Tensor:
        assert self.ledger is not None
        self.ledger.charge(draws)
        return _draw_logits(self.defense, candidate, draws).mean(0)


@dataclass
class PerDrawQueryOracle:
    """Return every response separately while charging every draw."""

    defense: Any
    ledger: QueryLedger | None = None

    def __post_init__(self) -> None:
        if self.ledger is None:
            self.ledger = QueryLedger()

    def query(self, candidate: torch.Tensor, draws: int) -> torch.Tensor:
        assert self.ledger is not None
        self.ledger.charge(draws)
        return _draw_logits(self.defense, candidate, draws)
