"""Differential tests against the user's archived defense and attack programs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import original_symbols
from torch import nn

from topk_fidelity import CleanReferenceTopKDefense, DefenseConfig
from topk_fidelity.attacks import run_square_batch
from topk_fidelity.attacks.objectives import exact_set_attack_margin
from topk_fidelity.config import AdaptiveConfig, RacingConfig
from topk_fidelity.models import freeze_model


class SmallNetwork(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 4 * 4, 12), nn.Tanh(), nn.Linear(12, classes)
        )

    def forward(self, x):
        return self.network(x)


@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("steps,sigma", [(0, 0.0), (2, 0.0), (0, 0.02), (2, 0.02)])
def test_defense_matches_original_logits_and_objective_gradients(k, steps, sigma):
    original = original_symbols(
        "evaluate_topk_confirmatory_holdout_cifar10.py",
        ["ranked_topk", "TopKTargetedGaussianDefense"],
    )
    torch.manual_seed(24)
    reference = freeze_model(SmallNetwork())
    anchor = freeze_model(SmallNetwork())
    images = torch.rand(3, 3, 4, 4)
    images[:, :, 0, 0] = 0
    images[:, :, -1, -1] = 1
    config = DefenseConfig(top_k=k, alpha=0.005, epsilon=0.03, steps=steps, sigma=sigma)
    old = original.TopKTargetedGaussianDefense(
        reference,
        anchor,
        anchor,
        k,
        config.alpha,
        config.epsilon,
        config.steps,
        config.sigma,
        config.top1_weight,
        config.set_weight,
    )
    new = CleanReferenceTopKDefense(reference, anchor, anchor, config)
    target = new.reference_set(images)
    variable = anchor(images).detach().requires_grad_(True)
    old_score = old.objective(variable, target)
    new_score = new.objective(variable, target)
    torch.testing.assert_close(old_score, new_score, rtol=0, atol=0)
    old_gradient = torch.autograd.grad(old_score.sum(), variable)[0]
    new_gradient = torch.autograd.grad(new_score.sum(), variable)[0]
    torch.testing.assert_close(old_gradient, new_gradient, rtol=0, atol=0)
    # Reset the SAME RNG before each stochastic implementation.
    torch.manual_seed(99)
    with torch.no_grad():
        expected = old(images)
    torch.manual_seed(99)
    with torch.no_grad():
        actual = new(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in anchor.parameters())


class NoisyResponse:
    """A reproducible full-logit oracle with both original interface routes."""

    def __init__(self, sigma):
        self.sigma = sigma

    def actual(self, x):
        feature = x.flatten(1)[:, :6]
        return feature + self.sigma * torch.randn_like(feature)

    def __call__(self, x):
        return self.actual(x)

    def draw_logits(self, x, draws):
        return torch.stack([self.actual(x) for _ in range(draws)])


@pytest.mark.parametrize("attack", ["SquareAdaptive", "SquareRacing"])
@pytest.mark.parametrize("sigma", [0.0, 0.07])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_generic_attacks_match_original_candidates_and_ledger(
    original_core, attack, sigma, batch_size
):
    torch.manual_seed(4)
    images = torch.rand(batch_size, 3, 4, 4)
    target = images.flatten(1)[:, :6].topk(2, dim=1).indices
    adaptive = AdaptiveConfig().as_dict()
    racing = RacingConfig(
        pool_size=3, race_interval=2, search_eot=2, refresh_eot=2, race_eot=2
    ).as_dict()
    kwargs = dict(
        attack=attack,
        defense=NoisyResponse(sigma),
        clean=images,
        reference_topk=target,
        eps=0.08,
        api_budget=77,
        p_init=0.3,
        adaptive_config=adaptive,
        racing_config=racing,
        proposal_seed=32,
        oracle_seed=38,
    )
    expected, old_details = original_core.run_square_batch(**kwargs)
    actual, new_details = run_square_batch(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert new_details == old_details
    assert new_details["api_queries"] == 77
    assert (actual - images).abs().max() <= 0.080001
    assert torch.all((actual >= 0) & (actual <= 1))


@pytest.mark.parametrize("sigma", [0.0, 0.07])
def test_cifar10_adaptive_preserves_zero_refresh_and_sequential_mean(
    original_c10, sigma
):
    torch.manual_seed(5)
    images = torch.rand(3, 3, 4, 4)
    target = images.flatten(1)[:, :6].topk(2, dim=1).indices
    refresh = 1 if sigma else 0
    adaptive = AdaptiveConfig(refresh_eot=refresh).as_dict()
    defense = NoisyResponse(sigma)
    oracle = original_c10.VariableQueryOracle(SimpleNamespace(actual=defense.actual))
    torch.manual_seed(38)
    expected, old_details = original_c10.square_adaptive_attack_budget(
        oracle,
        images,
        lambda z: original_c10.exact_set_attack_margin(z, target),
        0.08,
        77,
        0.3,
        1,
        refresh,
        5,
        0.20,
        sigma > 0,
        np.random.default_rng(32),
    )
    actual, details = run_square_batch(
        attack="SquareAdaptive",
        defense=defense,
        clean=images,
        reference_topk=target,
        eps=0.08,
        api_budget=77,
        p_init=0.3,
        adaptive_config=adaptive,
        racing_config=RacingConfig().as_dict(),
        proposal_seed=32,
        oracle_seed=38,
        mean_logit_reduction="sequential_sum",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert {key: details[key] for key in old_details} == old_details


@pytest.mark.parametrize("k", [2, 5])
def test_metric_values_match_original_and_hidden_failure_is_joint(k):
    original = original_symbols(
        "evaluate_topk_prior_baseline_attack_cifar10.py",
        [
            "ranked_topk",
            "sorted_set",
            "set_membership",
            "boundary_margin",
            "rank_weights",
            "MetricBatch",
            "compute_metric_batch",
        ],
    )
    from topk_fidelity.metrics import compute_metric_batch

    torch.manual_seed(5)
    logits, reference = torch.randn(7, 9), torch.randn(7, 9)
    labels = torch.arange(7)
    old = original.compute_metric_batch(logits, labels, reference, k, 0.5)
    new = compute_metric_batch(logits, labels, reference, k=k)
    for key in set(old.values) & set(new.values):
        torch.testing.assert_close(new.values[key], old.values[key], rtol=0, atol=0)
    expected_h = new.values["reference_top1_match"] * (1 - new.values["exact_set"])
    torch.testing.assert_close(
        new.values["hidden_topk_failure"], expected_h, rtol=0, atol=0
    )


def test_top1_agreement_and_exact_set_are_independent_events():
    from topk_fidelity.metrics import compute_metric_batch

    reference = torch.tensor([[5.0, 4.0, 0.0], [5.0, 4.0, 0.0]])
    predictions = torch.tensor([[5.0, 0.0, 4.0], [4.0, 5.0, 0.0]])
    batch = compute_metric_batch(predictions, torch.tensor([0, 0]), reference, k=2)
    assert batch.values["reference_top1_match"].tolist() == [1, 0]
    assert batch.values["exact_set"].tolist() == [0, 1]
    assert batch.values["hidden_topk_failure"].tolist() == [1, 0]
    assert exact_set_attack_margin(predictions, reference.topk(2).indices).tolist() == [
        -4,
        4,
    ]


def test_actual_cifar10_top2_wrapper_matches(original_c10):
    torch.manual_seed(15)
    reference = freeze_model(SmallNetwork(classes=10))
    anchor = freeze_model(SmallNetwork(classes=10))
    images = torch.rand(3, 3, 4, 4)
    config = DefenseConfig(2, 0.005, 0.03, 2, 0.02144277648925781)
    original = original_c10.base.TargetedGaussianDefense(
        reference,
        anchor,
        anchor,
        config.alpha,
        config.epsilon,
        config.steps,
        config.sigma,
        config.top1_weight,
        config.set_weight,
    )
    refactored = CleanReferenceTopKDefense(reference, anchor, anchor, config)
    torch.manual_seed(43)
    expected = original(images)
    torch.manual_seed(43)
    actual = refactored(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sigma", [0.0, 0.07])
def test_cifar10_racing_matches_original(original_c10, sigma):
    torch.manual_seed(5)
    images = torch.rand(3, 3, 4, 4)
    target = images.flatten(1)[:, :6].topk(2, dim=1).indices
    defense = NoisyResponse(sigma)
    racing = RacingConfig(
        pool_size=3, race_interval=2, search_eot=2, refresh_eot=2, race_eot=2
    )

    def score(logits):
        if sigma:
            return original_c10.per_draw_exact_set_survival_score(logits, target, 0.1)
        return original_c10.exact_set_attack_margin(logits[0], target)

    torch.manual_seed(38)
    expected, old_details = original_c10.square_racing_attack_budget(
        original_c10.PerDrawQueryOracle(SimpleNamespace(actual=defense.actual)),
        images,
        score,
        0.08,
        77,
        0.3,
        2 if sigma else 1,
        2 if sigma else 0,
        3 if sigma else 1,
        2 if sigma else 78,
        2 if sigma else 0,
        0.25,
        sigma > 0,
        0.1,
        np.random.default_rng(32),
    )
    actual, details = run_square_batch(
        attack="SquareRacing",
        defense=defense,
        clean=images,
        reference_topk=target,
        eps=0.08,
        api_budget=77,
        p_init=0.3,
        adaptive_config=AdaptiveConfig().as_dict(),
        racing_config=racing.as_dict(),
        proposal_seed=32,
        oracle_seed=38,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert details == old_details
