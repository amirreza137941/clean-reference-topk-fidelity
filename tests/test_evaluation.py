"""Protect per-response scoring, endpoint separation and request semantics."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from topk_fidelity import CleanReferenceTopKDefense, DefenseConfig, evaluate_responses
from topk_fidelity.query import PerDrawQueryOracle, QueryLedger


class AlternatingResponse:
    def __init__(self):
        self.position = 0

    def __call__(self, images):
        rows = [
            [5.0, 4.0, 3.0],  # exact set {0,1}
            [5.0, -100.0, 4.0],  # winner stays 0, set changes to {0,2}
        ]
        result = torch.tensor([rows[self.position % 2]])
        self.position += 1
        return result


def test_endpoint_counts_each_response_not_mean_logits_and_is_chunk_invariant():
    images, labels = torch.zeros(1, 1), torch.tensor([0])
    reference = torch.tensor([[5.0, 4.0, 0.0]])
    for chunk_size in (1, 3, 8):
        values = evaluate_responses(
            AlternatingResponse(),
            images,
            labels,
            reference,
            k=2,
            draws=8,
            chunk_size=chunk_size,
        )
        assert values["exact_set"].item() == 0.5
        assert values["hidden_topk_failure"].item() == 0.5
        assert values["reference_top1_match"].item() == 1.0


def test_budget_limit_blocks_calls_before_invoking_the_defense():
    defense = AlternatingResponse()
    ledger = QueryLedger(limit=3)
    oracle = PerDrawQueryOracle(defense, ledger)
    oracle.query(torch.zeros(1, 1), 2)
    with pytest.raises(RuntimeError, match="budget"):
        oracle.query(torch.zeros(1, 1), 2)
    assert defense.position == 2
    assert ledger.api_queries_per_example == 2
    assert ledger.logical_calls == 1


class CountingReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x


def test_online_reference_is_fixed_inside_request_and_recomputed_next_time():
    reference = CountingReference()
    anchor = nn.Identity()
    defense = CleanReferenceTopKDefense(
        reference,
        anchor,
        anchor,
        DefenseConfig(top_k=2, alpha=0.02, epsilon=0.05, steps=4, sigma=0.0),
    )
    first = torch.tensor([[0.8, 0.7, 0.1]])
    second = torch.tensor([[0.1, 0.7, 0.8]])
    defense(first)
    assert reference.calls == 1
    defense(second)
    assert reference.calls == 2
    assert defense.reference_set(second).tolist() == [[2, 1]]


def test_inference_mode_fails_with_actionable_gradient_message():
    model = nn.Identity()
    defense = CleanReferenceTopKDefense(
        model,
        model,
        model,
        DefenseConfig(top_k=2, alpha=0.02, epsilon=0.05, steps=2, sigma=0.0),
    )
    with torch.inference_mode(), pytest.raises(RuntimeError, match="input gradients"):
        defense(torch.tensor([[0.8, 0.7, 0.1]]))
