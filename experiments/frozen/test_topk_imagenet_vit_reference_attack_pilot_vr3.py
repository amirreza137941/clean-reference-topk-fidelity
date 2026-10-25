#!/usr/bin/env python3
"""Fail-fast tests for the ImageNet ViT-reference attack pilot VR3."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import topk_imagenet_vit_reference_attack_pilot_vr3_common as common
import topk_imagenet_vit_reference_square_attack_core_vr3 as core
from search_topk_imagenet_vit_reference_attack_pilot_vr3 import (
    FrozenDefense,
    add_payload_hash,
    identity,
    validate_candidate_payload,
)


def test_pure_protocol_rules() -> None:
    rows = [
        {
            "partition": "attack_pilot", "target": index,
            "class_name": f"n{index:08d}", "within_class_slot": 0,
            "relative_path": f"val/image_{index:05d}.JPEG",
        }
        for index in range(1000)
    ]
    first = common.select_pilot_rows(rows)
    second = common.select_pilot_rows(list(reversed(rows)))
    assert first == second and len(first) == 100
    assert len({row["target"] for row in first}) == 100
    assert common.method_spec("robust").stochastic is False
    assert common.method_spec("noise_only").stochastic is True
    assert common.method_spec("full").strength_steps == 4
    assert common.PRIMARY_K == 5 and common.EPSILON_PX == 4.0


def test_query_plans() -> None:
    for stochastic in (False, True):
        adaptive = core.square_adaptive_query_plan(
            common.API_BUDGET,
            common.ADAPTIVE_CONFIG["search_eot"],
            common.ADAPTIVE_CONFIG["refresh_eot"],
            common.ADAPTIVE_CONFIG["confirm_candidates"],
            common.ADAPTIVE_CONFIG["confirm_fraction"], stochastic,
        )
        assert adaptive["search_queries"] + sum(adaptive["confirmation_draws"]) == 5000
        cfg = common.RACING_CONFIG
        if stochastic:
            args = (cfg["search_eot"], cfg["refresh_eot"], cfg["pool_size"], cfg["race_interval"], cfg["race_eot"])
        else:
            args = (1, 0, 1, common.API_BUDGET + 1, 0)
        racing = core.square_racing_query_plan(
            common.API_BUDGET, *args, cfg["confirm_fraction"], stochastic
        )
        assert racing["search_queries"] + sum(racing["confirmation_draws"]) == 5000
    assert core.square_reference_p_selection(0.05, 0, 5000) == 0.05
    assert core.square_reference_p_selection(0.05, 4999, 5000) == 0.05 / 512


class TinyModel(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        weights = torch.linspace(-1.0, 1.0, 3000).view(3, 1000)
        self.register_buffer("weights", weights)
        self.offset = float(offset)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = images.mean(dim=(2, 3))
        return features @ self.weights + self.offset * torch.linspace(0, 1, 1000, device=images.device)


def test_attack_primitives_and_strengthening() -> None:
    clean = torch.rand(2, 3, 8, 8)
    rng = np.random.default_rng(7)
    initialized = core.square_vertical_stripe_initialization(clean, 4 / 255, rng)
    proposed = core.square_reference_linf_proposal(clean, initialized, 4 / 255, 0.05, rng)
    assert proposed.shape == clean.shape
    assert float((proposed - clean).abs().max()) <= 4 / 255 + 1e-7
    standard, robust = TinyModel(0.2), TinyModel(-0.1)
    spec = common.MethodSpec(
        name="tiny", sigma=0.01, strength_steps=1,
        strength_alpha_px=1.0, strength_epsilon_px=4.0,
        top1_weight=1.0, set_weight=4.0,
    )
    defense = FrozenDefense(spec, standard, robust, torch.device("cpu"))
    base = defense.base(clean)
    draws = defense.draw_logits(clean, 3, chunk_size=2)
    assert base.shape == clean.shape and draws.shape == (3, 2, 1000)
    assert torch.isfinite(draws).all()


def test_candidate_contract() -> None:
    protocol = {"protocol_hash": "f" * 64}
    expected = identity(protocol, "robust", "SquareAdaptive", common.SEARCH_SEEDS[0], "p" * 64)
    clean = torch.rand(2, 3, 224, 224)
    payload = {
        **expected,
        "relative_paths": ["val/a.JPEG", "val/b.JPEG"],
        "clean": clean,
        "adversarial": (clean + 1 / 255).clamp(0, 1),
        "labels": torch.tensor([1, 2]),
        "pilot_positions": torch.tensor([0, 1]),
        "reference_top5": torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]),
        "query_details": {"api_queries": 5000},
    }
    add_payload_hash(payload)
    validate_candidate_payload(payload, expected, [0, 1], payload["relative_paths"])


def test_live_parent_and_protocol() -> None:
    parent = common.validate_parent(common.DEFAULT_R0_DIR, common.DEFAULT_CALIBRATION_DIR, common.DEFAULT_CLEAN_DIR)
    payload = common.build_protocol_payload(
        parent=parent,
        data_root=common.DEFAULT_DATA_ROOT,
        output_root=common.DEFAULT_OUTPUT_ROOT,
        core_path=common.DEFAULT_ATTACK_CORE,
    )
    common.validate_protocol(payload)
    assert payload["confirmatory_outcomes_read"] is False
    assert payload["paper_claim_authorized"] is False
    assert payload["architecture_roles"] == {
        "standard_reference": "torchvision ViT-B/16",
        "defended_backbone": "torchvision ResNet-50",
        "cross_architecture_reference": True,
    }
    assert len(common.select_pilot_rows(parent["split_rows"])) == 100


def main() -> None:
    tests = (
        test_pure_protocol_rules,
        test_query_plans,
        test_attack_primitives_and_strengthening,
        test_candidate_contract,
        test_live_parent_and_protocol,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK", flush=True)
    print(f"All {len(tests)} ImageNet ViT-reference VR3 tests passed.", flush=True)


if __name__ == "__main__":
    main()
