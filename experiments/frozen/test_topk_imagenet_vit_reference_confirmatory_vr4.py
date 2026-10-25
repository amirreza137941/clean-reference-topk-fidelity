#!/usr/bin/env python3
"""Preflight tests for the frozen ImageNet ViT-reference VR4 confirmatory protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

import topk_imagenet_vit_reference_confirmatory_vr4_common as common
import topk_imagenet_vit_reference_square_attack_core_vr3 as core
from audit_topk_imagenet_vit_reference_confirmatory_vr4 import expected_query_details
from prepare_topk_imagenet_vit_reference_attack_pilot_vr3 import (
    validate_input_payload as validate_vr3_input,
)
from search_topk_imagenet_vit_reference_attack_pilot_vr3 import FrozenDefense
from search_topk_imagenet_vit_reference_confirmatory_vr4 import (
    add_payload_hash,
    atomic_torch_save,
    candidate_identity,
    validate_candidate_payload,
)
import calibrate_topk_imagenet_vit_reference_vr1 as vr1


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
        "robust": "SquareAdaptive",
        "full_sigma0": "SquareRacing",
        "noise_only": "SquareAdaptive",
        "full": "SquareAdaptive",
    }
    assert common.SEARCH_SEEDS == (474091, 585101, 696113)
    assert set(common.SEARCH_SEEDS).isdisjoint(common.PILOT_SEARCH_SEEDS)
    assert common.PRIMARY_K == 5 and common.SECONDARY_K == 2
    assert common.API_BUDGET == 5000 and common.EPSILON_PX == 4.0
    assert common.REEVAL_DRAWS == 1000
    assert common.EXPECTED_VR3_PROTOCOL_HASH == (
        "1ecd17d0945decd9d3dc6c4a829fb44ee68ebbc297a2937a643de528c0704330"
    )
    assert common.EXPECTED_VR3_SELECTION_HASH == (
        "ebe8acd871f8fc4c49f6dd047fa0b22fdbb90b0a503f5a76cddb1dd303ec40cb"
    )
    assert common.EXPECTED_CONFIRMATORY_ROWS_SHA256 == (
        "b4fec7303ad50ee992c65ae2cea4f4f69c0912f69a44f7cafc600b61562114a1"
    )


def test_vr3_freeze_and_outcome_blind_split() -> None:
    parent = common.load_parent(common.DEFAULT_R0_DIR, common.DEFAULT_CALIBRATION_DIR, common.DEFAULT_CLEAN_DIR)
    info = common.load_vr3_freeze(common.DEFAULT_VR3_ROOT, parent)
    rows = common.canonical_confirmatory_rows(parent)
    assert info["protocol_hash"] == common.EXPECTED_VR3_PROTOCOL_HASH
    assert info["selection_hash"] == common.EXPECTED_VR3_SELECTION_HASH
    assert len(rows) == len({row["relative_path"] for row in rows}) == 1000
    assert {row["target"] for row in rows} == set(range(1000))
    pilot_paths = {row["relative_path"] for row in parent["pilot_pool"]}
    assert pilot_paths.isdisjoint(row["relative_path"] for row in rows)
    assert info["protocol"]["confirmatory_outcomes_read"] is False


def test_protocol_build_and_analysis_freeze() -> None:
    parent = common.load_parent(common.DEFAULT_R0_DIR, common.DEFAULT_CALIBRATION_DIR, common.DEFAULT_CLEAN_DIR)
    info = common.load_vr3_freeze(common.DEFAULT_VR3_ROOT, parent)
    with tempfile.TemporaryDirectory() as temporary:
        payload = common.build_protocol_payload(
            parent=parent,
            vr3_info=info,
            data_root=common.DEFAULT_DATA_ROOT,
            output_root=Path(temporary) / "vr4",
        )
    assert common.validate_protocol(payload) == payload["protocol_hash"]
    assert payload["method_attack"] == common.METHOD_ATTACK
    assert payload["analysis_plan"]["primary_comparison"] == common.PRIMARY_COMPARISON
    assert payload["analysis_plan"]["no_attack_family_selection"] is True
    assert payload["analysis_plan"]["no_seed_selection"] is True
    assert payload["confirmatory_outcomes_read_before_protocol_freeze"] is False
    assert payload["architecture_roles"] == {
        "standard_reference": "torchvision ViT-B/16",
        "defended_backbone": "torchvision ResNet-50",
        "cross_architecture_reference": True,
    }
    assert payload["experiment_scope"].endswith("not a robust-ViT evaluation")
    assert payload["standard_reference_adversarially_robust"] is False
    assert payload["robust_vit_claim_authorized"] is False


def test_frozen_query_plans() -> None:
    for method in common.METHODS:
        details = expected_query_details(method)
        assert int(details["api_queries"]) == common.API_BUDGET
        assert int(details["search_steps"]) > 0
        assert int(details["confirmation_queries"]) > 0
        if common.METHOD_ATTACK[method] == "SquareAdaptive":
            assert details["score_objective"] == common.ADAPTIVE_CONFIG["objective"]
        elif common.method_spec(method).stochastic:
            assert details["score_objective"] == common.RACING_CONFIG["objective"]
        else:
            assert details["score_objective"] == "deterministic_exact_set_margin"


def test_square_core_and_candidate_roundtrip() -> None:
    clean = torch.full((2, 3, 224, 224), 0.5)
    reference = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    eps = common.EPSILON_PX / 255.0
    initialized = core.square_vertical_stripe_initialization(
        clean, eps, np.random.default_rng(1)
    )
    proposed = core.square_reference_linf_proposal(
        clean, initialized, eps, 0.05, np.random.default_rng(2)
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
        assert int(details["api_queries"]) == budget
        assert float((candidate - clean).abs().max()) <= eps + 1e-7

    protocol = {"protocol_hash": "a" * 64}
    identity = candidate_identity(
        protocol,
        "full",
        common.SEARCH_SEEDS[0],
        "b" * 64,
        common.tensor_sha256(clean),
    )
    payload = {
        **identity,
        "relative_paths": ["n00000001/a.JPEG", "n00000002/b.JPEG"],
        "adversarial": clean.clone(),
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "confirmatory_positions": torch.tensor([0, 1], dtype=torch.long),
        "reference_top5": torch.tensor(
            [[0, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.long
        ),
        "query_details": {"api_queries": common.API_BUDGET},
        "candidate_schema": "shared_clean_manifest_plus_adversarial_tensor_v1",
    }
    add_payload_hash(payload)
    validate_candidate_payload(
        payload,
        identity,
        [0, 1],
        payload["relative_paths"],
        payload["labels"],
        payload["reference_top5"],
        clean=clean,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "candidate.pt"
        atomic_torch_save(path, payload)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        validate_candidate_payload(
            loaded,
            identity,
            [0, 1],
            payload["relative_paths"],
            payload["labels"],
            payload["reference_top5"],
            clean=clean,
        )


def test_models_using_only_open_vr3_pilot() -> None:
    """Never read VR4 images during the self-test."""

    parent = common.load_parent(common.DEFAULT_R0_DIR, common.DEFAULT_CALIBRATION_DIR, common.DEFAULT_CLEAN_DIR)
    info = common.load_vr3_freeze(common.DEFAULT_VR3_ROOT, parent)
    pilot_path = Path(info["vr3_root"]) / "pilot_inputs.pt"
    pilot = torch.load(pilot_path, map_location="cpu", weights_only=False)
    validate_vr3_input(pilot, info["protocol"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    standard, robust_model, _audit = vr1.load_models(
        Path(info["protocol"]["checkpoints"]["standard"]["path"]),
        Path(info["protocol"]["checkpoints"]["robust"]["path"]),
        device,
    )
    images = pilot["clean"][:1].to(device)
    robust = FrozenDefense(common.method_spec("robust"), standard, robust_model, device)
    full = FrozenDefense(common.method_spec("full"), standard, robust_model, device)
    robust_logits = robust.draw_logits(images, 1)
    full_logits = full.draw_logits(images, 1)
    assert robust_logits.shape == (1, 1, 1000)
    assert full_logits.shape == (1, 1, 1000)
    assert torch.isfinite(robust_logits).all() and torch.isfinite(full_logits).all()


def main() -> None:
    tests = (
        test_frozen_constants,
        test_vr3_freeze_and_outcome_blind_split,
        test_protocol_build_and_analysis_freeze,
        test_frozen_query_plans,
        test_square_core_and_candidate_roundtrip,
        test_models_using_only_open_vr3_pilot,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK", flush=True)
    print("All ImageNet ViT-reference confirmatory VR4 tests passed.", flush=True)


if __name__ == "__main__":
    main()
