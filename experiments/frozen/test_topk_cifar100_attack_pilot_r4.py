#!/usr/bin/env python3
"""Preflight/self-tests for CIFAR-100 adaptive attack pilot R4."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import calibrate_topk_cifar100_actual_defense_r1 as r1
import topk_cifar100_attack_pilot_r4_common as common
import topk_cifar100_square_attack_core_r4 as core
from search_topk_cifar100_attack_pilot_r4 import (
    FrozenDefense,
    add_payload_hash,
    atomic_torch_save,
    load_models,
    payload_identity,
    validate_payload,
)


class ToyDefense:
    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def actual(self, x: torch.Tensor) -> torch.Tensor:
        value = x.flatten(1).mean(1)
        if self.sigma > 0:
            value = value + self.sigma * torch.randn_like(value)
        return torch.stack((value, -value, torch.zeros_like(value)), dim=1)


def test_frozen_protocol_constants() -> None:
    assert common.METHODS == ("robust", "full_sigma0", "noise_only", "full")
    assert common.ATTACKS == ("SquareAdaptive", "SquareRacing")
    assert len(common.SEARCH_SEEDS) == 3
    assert len(common.CONFIRMATORY_SEARCH_SEEDS) == 3
    assert set(common.SEARCH_SEEDS).isdisjoint(common.CONFIRMATORY_SEARCH_SEEDS)
    assert common.TOP_K == 2 and common.API_BUDGET == 5000
    assert common.EPSILON_PX == 2.0 and common.REEVAL_DRAWS == 1000
    assert common.SEARCH_BATCH_SIZE_BY_METHOD == {
        "robust": 20,
        "noise_only": 20,
        "full_sigma0": 5,
        "full": 5,
    }
    assert common.method_spec("full").stochastic
    assert not common.method_spec("full_sigma0").stochastic


def test_query_plans() -> None:
    for stochastic in (False, True):
        adaptive = core.square_adaptive_query_plan(
            common.API_BUDGET,
            int(common.ADAPTIVE_CONFIG["search_eot"]),
            int(common.ADAPTIVE_CONFIG["refresh_eot"]),
            int(common.ADAPTIVE_CONFIG["confirm_candidates"]),
            float(common.ADAPTIVE_CONFIG["confirm_fraction"]),
            stochastic,
        )
        assert adaptive["search_queries"] + sum(adaptive["confirmation_draws"]) == 5000
        if stochastic:
            racing_args = (
                int(common.RACING_CONFIG["search_eot"]),
                int(common.RACING_CONFIG["refresh_eot"]),
                int(common.RACING_CONFIG["pool_size"]),
                int(common.RACING_CONFIG["race_interval"]),
                int(common.RACING_CONFIG["race_eot"]),
            )
        else:
            racing_args = (1, 0, 1, common.API_BUDGET + 1, 0)
        racing = core.square_racing_query_plan(
            common.API_BUDGET,
            *racing_args,
            float(common.RACING_CONFIG["confirm_fraction"]),
            stochastic,
        )
        assert racing["search_queries"] + sum(racing["confirmation_draws"]) == 5000
        assert racing["race_queries"] == racing["race_rounds"] * racing_args[2] * racing_args[4]


def test_square_proposals_and_toy_attacks() -> None:
    clean = torch.full((3, 3, 32, 32), 0.5)
    reference = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.long)
    eps = 2.0 / 255.0
    initialized = core.square_vertical_stripe_initialization(
        clean, eps, np.random.default_rng(1)
    )
    proposed = core.square_reference_linf_proposal(
        clean, initialized, eps, 0.3, np.random.default_rng(2)
    )
    assert proposed.shape == clean.shape
    assert float((proposed - clean).abs().max()) <= eps + 1e-7
    adaptive_config = {
        "search_eot": 1,
        "refresh_eot": 1,
        "confirm_candidates": 5,
        "confirm_fraction": 0.2,
        "objective": "exact_set_margin_of_mean_logits",
    }
    racing_config = {
        "search_eot": 2,
        "refresh_eot": 2,
        "pool_size": 4,
        "race_interval": 3,
        "race_eot": 1,
        "confirm_fraction": 0.2,
        "score_temperature": 0.1,
        "objective": "mean_per_draw_sigmoid_exact_set_margin",
    }
    for attack, defense, budget in (
        ("SquareAdaptive", ToyDefense(0.01), 60),
        ("SquareRacing", ToyDefense(0.01), 80),
        ("SquareRacing", ToyDefense(0.0), 40),
    ):
        candidate, details = core.run_square_batch(
            attack=attack,
            defense=defense,
            clean=clean,
            reference_topk=reference,
            eps=eps,
            api_budget=budget,
            p_init=0.3,
            adaptive_config=adaptive_config,
            racing_config=racing_config,
            proposal_seed=11,
            oracle_seed=12,
        )
        assert details["api_queries"] == budget
        assert float((candidate - clean).abs().max()) <= eps + 1e-7
        assert 0 <= float(candidate.min()) <= float(candidate.max()) <= 1


def test_candidate_schema_roundtrip() -> None:
    clean = torch.full((2, 3, 32, 32), 0.5)
    adversarial = clean.clone()
    identity = payload_identity(
        protocol_hash="a" * 64,
        method="full",
        attack="SquareAdaptive",
        search_seed=common.SEARCH_SEEDS[0],
    )
    payload = {
        **identity,
        "clean": clean,
        "adversarial": adversarial,
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "sample_indices": torch.tensor([10, 20], dtype=torch.long),
        "reference_top2": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        "query_details": {"api_queries": common.API_BUDGET},
    }
    add_payload_hash(payload)
    validate_payload(payload, identity, [10, 20])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "candidate.pt"
        atomic_torch_save(path, payload)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        validate_payload(loaded, identity, [10, 20])
        assert "clean" in loaded and "adversarial" in loaded


def test_parent_dataset_models_and_strengthening() -> None:
    parent = common.load_parent_authorization(common.DEFAULT_PARENT_DIR)
    root = r1.r2_anchor.resolve_torchvision_root(common.DEFAULT_DATA_DIR)
    dataset = torchvision.datasets.CIFAR100(
        root=str(root), train=False, download=False, transform=transforms.ToTensor()
    )
    common.validate_dataset_against_parent(dataset, parent)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protocol_stub = {
        "checkpoints": parent["handoff"]["checkpoints"],
    }
    standard, anchor = load_models(protocol_stub, device)
    indices = parent["pilot_indices"][:2]
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    robust = FrozenDefense(common.method_spec("robust"), standard, anchor, device)
    full = FrozenDefense(common.method_spec("full"), standard, anchor, device)
    with torch.no_grad():
        robust_logits = robust.actual(images)
    full_logits = full.actual(images)
    assert robust_logits.shape == (2, 100)
    assert full_logits.shape == (2, 100)
    assert torch.isfinite(robust_logits).all() and torch.isfinite(full_logits).all()


def main() -> None:
    tests = (
        test_frozen_protocol_constants,
        test_query_plans,
        test_square_proposals_and_toy_attacks,
        test_candidate_schema_roundtrip,
        test_parent_dataset_models_and_strengthening,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK", flush=True)
    print("All CIFAR-100 attack-pilot R4 tests passed.", flush=True)


if __name__ == "__main__":
    main()
