import pytest
import torch

from topk_fidelity.attacks.square import (
    exact_set_attack_margin,
    square_adaptive_query_plan,
    square_racing_query_plan,
    square_reference_p_selection,
)


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(0, 0.8), (11, 0.4), (51, 0.2), (201, 0.1), (9000, 0.8 / 512)],
)
def test_reference_p_schedule(iteration: int, expected: float) -> None:
    assert square_reference_p_selection(0.8, iteration, 10000) == pytest.approx(
        expected
    )


def test_exact_set_margin() -> None:
    logits = torch.tensor([[5.0, 4.0, 2.0, 1.0]])
    reference = torch.tensor([[0, 1]])
    assert exact_set_attack_margin(logits, reference).item() == 2.0


def test_adaptive_plan_spends_exact_budget() -> None:
    plan = square_adaptive_query_plan(
        api_budget=5000,
        search_eot=5,
        refresh_eot=5,
        confirm_candidates=5,
        confirm_fraction=0.2,
        stochastic_defense=True,
    )
    assert plan["search_queries"] + sum(plan["confirmation_draws"]) == 5000


def test_racing_plan_spends_exact_budget() -> None:
    plan = square_racing_query_plan(
        api_budget=5000,
        search_eot=5,
        refresh_eot=5,
        pool_size=5,
        race_interval=25,
        race_eot=5,
        confirm_fraction=0.2,
        stochastic_defense=True,
    )
    assert plan["search_queries"] + sum(plan["confirmation_draws"]) == 5000
