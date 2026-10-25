#!/usr/bin/env python3
"""Preflight/self-tests for frozen CIFAR-100 confirmatory R5."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import calibrate_topk_cifar100_actual_defense_r1 as r1
import topk_cifar100_confirmatory_r5_common as common
import topk_cifar100_square_attack_core_r4 as core
from audit_topk_cifar100_confirmatory_r5 import expected_query_details
from search_topk_cifar100_attack_pilot_r4 import (
    FrozenDefense,
    add_payload_hash,
    atomic_torch_save,
    load_models,
)
from search_topk_cifar100_confirmatory_r5 import payload_identity, validate_payload


class ToyDefense:
    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def actual(self, x: torch.Tensor) -> torch.Tensor:
        value = x.flatten(1).mean(1)
        if self.sigma > 0:
            value = value + self.sigma * torch.randn_like(value)
        return torch.stack((value, -value, torch.zeros_like(value)), dim=1)


def test_frozen_confirmatory_constants() -> None:
    assert common.METHODS == ("robust", "full_sigma0", "noise_only", "full")
    assert common.METHOD_ATTACK == {
        "robust": "SquareRacing",
        "full_sigma0": "SquareRacing",
        "noise_only": "SquareAdaptive",
        "full": "SquareAdaptive",
    }
    assert common.SEARCH_SEEDS == (71071, 82073, 93083)
    assert set(common.SEARCH_SEEDS).isdisjoint(common.PILOT_SEARCH_SEEDS)
    assert common.TOP_K == 2 and common.API_BUDGET == 5000
    assert common.EPSILON_PX == 2.0 and common.REEVAL_DRAWS == 1000
    assert common.PRIMARY_COMPARISON == "full_minus_robust_attacked_exact_set"
    assert common.EXPECTED_R4_PROTOCOL_HASH == (
        "492bb3a72ea5d67f94c5623dbdceeb26b07774446073e4b6f062f740b64da6e9"
    )
    assert common.EXPECTED_R4_SELECTION_HASH == (
        "acda4c5abb3774d4bffdc499260f2856d9facb41fa300fd2907b9c944ccdfa84"
    )


def test_r4_freeze_and_r3_partition() -> None:
    parent = common.load_parent_authorization(common.DEFAULT_PARENT_DIR)
    r4_info = common.load_r4_freeze(common.DEFAULT_R4_ROOT, parent)
    assert r4_info["protocol_hash"] == common.EXPECTED_R4_PROTOCOL_HASH
    assert r4_info["selection_hash"] == common.EXPECTED_R4_SELECTION_HASH
    confirmatory = [int(value) for value in parent["confirmatory_indices"]]
    pilot = [int(value) for value in parent["pilot_indices"]]
    assert len(confirmatory) == len(set(confirmatory)) == 1000
    assert set(confirmatory).isdisjoint(pilot)
    assert common.sequence_sha256(confirmatory) == common.EXPECTED_CONFIRMATORY_INDICES_SHA256


def test_protocol_build_and_analysis_freeze() -> None:
    parent = common.load_parent_authorization(common.DEFAULT_PARENT_DIR)
    r4_info = common.load_r4_freeze(common.DEFAULT_R4_ROOT, parent)
    with tempfile.TemporaryDirectory() as temporary:
        payload = common.build_protocol_payload(
            parent=parent,
            r4_info=r4_info,
            data_dir=common.DEFAULT_DATA_DIR,
            r2_root=common.DEFAULT_R2_ROOT,
            dev_dir=common.DEFAULT_DEV_DIR,
            output_root=Path(temporary) / "r5",
        )
    assert common.validate_protocol(payload) == payload["protocol_hash"]
    assert payload["method_attack"] == common.METHOD_ATTACK
    assert payload["analysis_plan"]["primary_comparison"] == common.PRIMARY_COMPARISON
    assert payload["analysis_plan"]["no_attack_family_selection"] is True
    assert payload["analysis_plan"]["no_seed_selection"] is True


def test_frozen_query_plans() -> None:
    for method in common.METHODS:
        details = expected_query_details(method)
        assert int(details["api_queries"]) == common.API_BUDGET
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
    toy_adaptive = dict(common.ADAPTIVE_CONFIG)
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
            adaptive_config=toy_adaptive,
            racing_config=toy_racing,
            proposal_seed=11,
            oracle_seed=12,
        )
        assert details["api_queries"] == budget
        assert float((candidate - clean).abs().max()) <= eps + 1e-7

    identity = payload_identity(
        protocol_hash="a" * 64,
        method="full",
        search_seed=common.SEARCH_SEEDS[0],
    )
    payload = {
        **identity,
        "clean": clean,
        "adversarial": clean.clone(),
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "sample_indices": torch.tensor([10, 20], dtype=torch.long),
        "reference_top2": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        "query_details": {"api_queries": common.API_BUDGET},
        "r4_selection_hash": common.EXPECTED_R4_SELECTION_HASH,
    }
    add_payload_hash(payload)
    validate_payload(payload, identity, [10, 20])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "candidate.pt"
        atomic_torch_save(path, payload)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        validate_payload(loaded, identity, [10, 20])


def test_models_without_confirmatory_outcome_access() -> None:
    """Use only already-open R4 pilot images for the model-interface preflight."""

    parent = common.load_parent_authorization(common.DEFAULT_PARENT_DIR)
    root = r1.r2_anchor.resolve_torchvision_root(common.DEFAULT_DATA_DIR)
    dataset = torchvision.datasets.CIFAR100(
        root=str(root), train=False, download=False, transform=transforms.ToTensor()
    )
    common.validate_dataset_against_parent(dataset, parent)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protocol_stub = {"checkpoints": parent["handoff"]["checkpoints"]}
    standard, anchor = load_models(protocol_stub, device)
    indices = parent["pilot_indices"][:2]
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    robust = FrozenDefense(common.method_spec("robust"), standard, anchor, device)
    full = FrozenDefense(common.method_spec("full"), standard, anchor, device)
    with torch.no_grad():
        robust_logits = robust.actual(images)
    full_logits = full.actual(images)
    assert robust_logits.shape == (2, 100) and full_logits.shape == (2, 100)
    assert torch.isfinite(robust_logits).all() and torch.isfinite(full_logits).all()


def main() -> None:
    tests = (
        test_frozen_confirmatory_constants,
        test_r4_freeze_and_r3_partition,
        test_protocol_build_and_analysis_freeze,
        test_frozen_query_plans,
        test_square_core_and_candidate_roundtrip,
        test_models_without_confirmatory_outcome_access,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK", flush=True)
    print("All CIFAR-100 confirmatory R5 tests passed.", flush=True)


if __name__ == "__main__":
    main()
