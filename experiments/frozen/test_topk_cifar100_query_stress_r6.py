#!/usr/bin/env python3
"""Preflight/self-tests for CIFAR-100 R6 query-budget stress."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import audit_topk_cifar100_query_stress_r6 as audit
import calibrate_topk_cifar100_actual_defense_r1 as r1
import topk_cifar100_query_stress_r6_common as common
import topk_cifar100_square_attack_core_r4 as core
from search_topk_cifar100_attack_pilot_r4 import (
    FrozenDefense,
    add_payload_hash,
    atomic_torch_save,
    load_models,
)
from search_topk_cifar100_query_stress_r6 import payload_identity, validate_payload


class ToyDefense:
    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def actual(self, x: torch.Tensor) -> torch.Tensor:
        value = x.flatten(1).mean(1)
        if self.sigma > 0:
            value = value + self.sigma * torch.randn_like(value)
        return torch.stack((value, -value, torch.zeros_like(value)), dim=1)


def test_frozen_constants() -> None:
    assert common.METHODS == ("robust", "full_sigma0", "noise_only", "full")
    assert common.METHOD_ATTACK == {
        "robust": "SquareRacing",
        "full_sigma0": "SquareRacing",
        "noise_only": "SquareAdaptive",
        "full": "SquareAdaptive",
    }
    assert common.BUDGETS == (5000, 20000)
    assert common.SEARCH_SEEDS == (111071, 122073, 133083)
    assert common.STRESS_EXAMPLES == 200 and common.STRESS_PER_CLASS == 2
    assert common.TOP_K == 2 and common.EPSILON_PX == 2.0
    assert common.REEVAL_DRAWS == 1000
    assert common.EXPECTED_R5_PROTOCOL_HASH == (
        "5cea83fc79fa5f1c1b4ee2e3674a87aa6a7172483d2f428e191ca2cdf953bfba"
    )
    assert common.EXPECTED_R5_FINAL_SUMMARY_HASH == (
        "4327b1e20c51bb1fa7b2a4c44c5d82d1a7a1b39e7a0c65f042821bb862b8d778"
    )


def test_r5_completion_and_disjoint_split() -> None:
    r5_info = common.load_r5_completion(common.DEFAULT_R5_ROOT)
    parent = common.r5.load_parent_authorization(common.DEFAULT_R3_PARENT_DIR)
    assert r5_info["protocol"]["parent_handoff_hash"] == parent["handoff_hash"]
    first = common.derive_stress_indices(parent)
    second = common.derive_stress_indices(parent)
    assert first == second
    common.validate_stress_indices(first, parent)
    assert len(first) == len(set(first)) == 200
    excluded = set(parent["pilot_indices"]) | set(parent["confirmatory_indices"])
    assert not (set(first) & excluded)
    labels, references = common.load_stress_references(parent, first)
    assert len(labels) == len(references) == 200
    assert all(len(reference) == len(set(reference)) == 2 for reference in references)


def test_protocol_build_and_freeze() -> None:
    common.validate_script_inventory()
    parent = common.r5.load_parent_authorization(common.DEFAULT_R3_PARENT_DIR)
    r5_info = common.load_r5_completion(common.DEFAULT_R5_ROOT)
    indices = common.derive_stress_indices(parent)
    with tempfile.TemporaryDirectory() as temporary:
        payload = common.build_protocol_payload(
            parent=parent,
            r5_info=r5_info,
            stress_indices=indices,
            data_dir=common.DEFAULT_DATA_DIR,
            r2_root=common.DEFAULT_R2_ROOT,
            dev_dir=common.DEFAULT_DEV_DIR,
            output_root=Path(temporary) / "r6",
        )
    assert common.validate_protocol(payload) == payload["protocol_hash"]
    assert payload["stress_indices"] == indices
    assert payload["analysis_plan"]["paper_primary_claim_authorized"] is False
    assert payload["analysis_plan"]["no_per_example_budget_selection"] is True


def test_exact_query_plans() -> None:
    for method in common.METHODS:
        for budget in common.BUDGETS:
            details = audit.expected_query_details(method, budget)
            assert int(details["api_queries"]) == budget
            assert int(details["search_steps"]) > 0
            assert int(details["confirmation_queries"]) > 0
            if common.METHOD_ATTACK[method] == "SquareAdaptive":
                assert details["score_objective"] == common.ADAPTIVE_CONFIG["objective"]
            else:
                assert details["score_objective"] == "deterministic_exact_set_margin"


def test_square_core_and_candidate_roundtrip() -> None:
    clean = torch.full((2, 3, 32, 32), 0.5)
    reference = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    eps = common.EPSILON_PX / 255.0
    initialized = core.square_vertical_stripe_initialization(
        clean, eps, np.random.default_rng(1)
    )
    proposed = core.square_reference_linf_proposal(
        clean, initialized, eps, 0.3, np.random.default_rng(2)
    )
    assert proposed.shape == clean.shape
    assert float((proposed - clean).abs().max()) <= eps + 1e-7
    toy_racing = dict(common.RACING_CONFIG)
    toy_racing.update(
        {
            "search_eot": 2,
            "refresh_eot": 2,
            "pool_size": 4,
            "race_interval": 3,
            "race_eot": 1,
        }
    )
    for attack, defense, budget in (
        ("SquareAdaptive", ToyDefense(0.01), 60),
        ("SquareRacing", ToyDefense(0.0), 40),
    ):
        candidate, details = core.run_square_batch(
            attack=attack,
            defense=defense,
            clean=clean,
            reference_topk=reference,
            eps=eps,
            api_budget=budget,
            p_init=common.SQUARE_P_INIT,
            adaptive_config=common.ADAPTIVE_CONFIG,
            racing_config=toy_racing,
            proposal_seed=11,
            oracle_seed=12,
        )
        assert details["api_queries"] == budget
        assert float((candidate - clean).abs().max()) <= eps + 1e-7

    identity = payload_identity(
        protocol_hash="a" * 64,
        method="full",
        budget=common.BUDGETS[0],
        search_seed=common.SEARCH_SEEDS[0],
    )
    payload = {
        **identity,
        "clean": clean,
        "adversarial": clean.clone(),
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "sample_indices": torch.tensor([10, 20], dtype=torch.long),
        "reference_top2": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        "query_details": {"api_queries": common.BUDGETS[0]},
        "r5_final_summary_hash": common.EXPECTED_R5_FINAL_SUMMARY_HASH,
    }
    add_payload_hash(payload)
    validate_payload(payload, identity, [10, 20])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "candidate.pt"
        atomic_torch_save(path, payload)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        validate_payload(loaded, identity, [10, 20])


def test_models_on_stress_inputs() -> None:
    parent = common.r5.load_parent_authorization(common.DEFAULT_R3_PARENT_DIR)
    indices = common.derive_stress_indices(parent)
    dataset_root = r1.r2_anchor.resolve_torchvision_root(common.DEFAULT_DATA_DIR)
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root), train=False, download=False, transform=transforms.ToTensor()
    )
    common.r5.validate_dataset_against_parent(dataset, parent)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protocol_stub = {"checkpoints": parent["handoff"]["checkpoints"]}
    standard, anchor = load_models(protocol_stub, device)
    images = torch.stack([dataset[index][0] for index in indices[:2]]).to(device)
    robust = FrozenDefense(common.method_spec("robust"), standard, anchor, device)
    full = FrozenDefense(common.method_spec("full"), standard, anchor, device)
    with torch.no_grad():
        robust_logits = robust.actual(images)
    full_logits = full.actual(images)
    assert robust_logits.shape == (2, 100) and full_logits.shape == (2, 100)
    assert torch.isfinite(robust_logits).all() and torch.isfinite(full_logits).all()


def main() -> None:
    tests = (
        test_frozen_constants,
        test_r5_completion_and_disjoint_split,
        test_protocol_build_and_freeze,
        test_exact_query_plans,
        test_square_core_and_candidate_roundtrip,
        test_models_on_stress_inputs,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK", flush=True)
    print("All CIFAR-100 R6 query-stress tests passed.", flush=True)


if __name__ == "__main__":
    main()
