import torch

from topk_fidelity.query import AveragedQueryOracle, PerDrawQueryOracle


class CountingDefense:
    def draw_logits(self, candidate: torch.Tensor, draws: int) -> torch.Tensor:
        parts = [
            torch.full((candidate.shape[0], 2), float(index)) for index in range(draws)
        ]
        return torch.stack(parts, dim=0)


def test_averaged_oracle_charges_every_draw() -> None:
    oracle = AveragedQueryOracle(CountingDefense())
    result = oracle.query(torch.zeros(4, 1), draws=3)
    assert torch.all(result == 1.0)
    assert oracle.ledger is not None
    assert oracle.ledger.logical_calls == 1
    assert oracle.ledger.api_queries_per_example == 3


def test_per_draw_oracle_preserves_draw_axis() -> None:
    oracle = PerDrawQueryOracle(CountingDefense())
    result = oracle.query(torch.zeros(2, 1), draws=5)
    assert result.shape == (5, 2, 2)
    assert oracle.ledger is not None
    assert oracle.ledger.api_queries_per_example == 5
