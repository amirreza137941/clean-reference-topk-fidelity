#!/usr/bin/env python3
"""Fast contract tests for the D1 Q=20k/EOT stress-test package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import topk_cifar100_square_attack_core_r4 as core
import topk_imagenet_posthoc_q20k_eot_d1_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--with-parent", action="store_true")
    return parser.parse_args()


def test_constants() -> None:
    assert common.METHODS == ("robust", "full")
    assert common.METHOD_ATTACK == {
        "robust": "SquareAdaptive",
        "full": "SquareAdaptive",
    }
    assert common.SEARCH_SEEDS == (807019, 918031, 1029043)
    assert len(set(common.SEARCH_SEEDS)) == 3
    assert not set(common.SEARCH_SEEDS).intersection(common.r5.SEARCH_SEEDS)
    assert common.DIAGNOSTIC_EXAMPLES == 100
    assert common.API_BUDGET == 20_000
    assert common.EPSILON_PX == 4.0
    assert common.REEVAL_DRAWS == 1000
    assert common.REEVAL_BATCH_SIZE == 4
    assert common.REEVAL_MC_CHUNK == 10
    assert common.BOOTSTRAP_RESAMPLES == 50_000
    assert common.EXPECTED_SUBSET_ROWS_SHA256 == (
        "516d2cfd84e4ec9431c619171fc4125b4d82b7302796051beee8e96d71f91aac"
    )
    assert common.FROZEN_DEFENSE == dict(common.r5.FROZEN_DEFENSE)
    assert common.method_spec("robust").sigma == 0.0
    full = common.method_spec("full")
    assert full.sigma == 0.025
    assert full.strength_steps == 4
    assert full.strength_alpha_px == 1.0
    assert full.strength_epsilon_px == 4.0
    assert full.top1_weight == 1.0
    assert full.set_weight == 4.0


def test_source_manifest() -> None:
    for name in (*common.SCRIPT_NAMES, *common.DEPENDENCY_NAMES):
        path = common.PROJECT_DIR / name
        assert path.is_file(), f"Missing required source: {path}"
    core_path = common.PROJECT_DIR / "topk_cifar100_square_attack_core_r4.py"
    assert common.file_sha256(core_path) == common.EXPECTED_ATTACK_CORE_SHA256


def test_subset_rule() -> None:
    rows = [
        {
            "confirmatory_position": position,
            "target": position,
            "class_name": f"class_{position:04d}",
            "relative_path": f"n{position:08d}/image_{position:04d}.JPEG",
        }
        for position in range(1000)
    ]
    selected = common.select_subset_rows(rows)
    reversed_selected = common.select_subset_rows(list(reversed(rows)))
    assert selected == reversed_selected
    assert len(selected) == common.DIAGNOSTIC_EXAMPLES
    assert [row["diagnostic_position"] for row in selected] == list(range(100))
    ranks = [row["selection_rank_sha256"] for row in selected]
    assert ranks == sorted(ranks)
    assert len({row["target"] for row in selected}) == 100
    assert common.subset_rows_sha256(selected) == common.subset_rows_sha256(
        reversed_selected
    )


def test_query_plans() -> None:
    robust = common.ADAPTIVE_CONFIG_BY_METHOD["robust"]
    robust_plan = core.square_adaptive_query_plan(
        common.API_BUDGET,
        robust["search_eot"],
        robust["refresh_eot"],
        robust["confirm_candidates"],
        robust["confirm_fraction"],
        False,
    )
    assert robust_plan == {
        "search_steps": 9997,
        "search_queries": 19995,
        "confirmation_queries": 5,
        "confirmation_draws": [1, 1, 1, 1, 1],
        "logical_calls": 20000,
    }
    assert robust_plan["search_queries"] + sum(
        robust_plan["confirmation_draws"]
    ) == common.API_BUDGET

    full = common.ADAPTIVE_CONFIG_BY_METHOD["full"]
    full_plan = core.square_adaptive_query_plan(
        common.API_BUDGET,
        full["search_eot"],
        full["refresh_eot"],
        full["confirm_candidates"],
        full["confirm_fraction"],
        True,
    )
    assert full_plan == {
        "search_steps": 1599,
        "search_queries": 15995,
        "confirmation_queries": 4005,
        "confirmation_draws": [801, 801, 801, 801, 801],
        "logical_calls": 3204,
    }
    assert full_plan["search_queries"] + sum(
        full_plan["confirmation_draws"]
    ) == common.API_BUDGET


def test_bootstrap() -> None:
    delta = np.linspace(-0.25, 0.75, common.DIAGNOSTIC_EXAMPLES)
    first = common.paired_bootstrap(delta, resamples=2000, seed=12345)
    second = common.paired_bootstrap(delta, resamples=2000, seed=12345)
    assert first == second
    assert abs(float(first["difference_pp"]) - 25.0) < 1e-12
    assert first["examples"] == common.DIAGNOSTIC_EXAMPLES
    assert first["resamples"] == 2000


def test_protocol(path: Path) -> None:
    payload = common.read_json(path.expanduser().resolve())
    common.validate_protocol(payload)
    assert payload["post_hoc"] is True
    assert payload["paper_claim_authorized"] is False
    assert payload["may_modify_parent_r5"] is False
    assert payload["no_training"] is True
    assert payload["no_defense_retuning"] is True
    assert payload["no_attack_family_selection"] is True
    assert payload["no_seed_selection"] is True
    assert payload["no_per_example_candidate_selection"] is True


def main() -> None:
    args = parse_args()
    test_constants()
    test_source_manifest()
    test_subset_rule()
    test_query_plans()
    test_bootstrap()
    if args.protocol_json is not None:
        test_protocol(args.protocol_json)
    if args.with_parent:
        parent = common.load_parent_r5(common.DEFAULT_R5_ROOT)
        assert parent["protocol_hash"] == common.EXPECTED_R5_PROTOCOL_HASH
        assert len(parent["subset"]) == common.DIAGNOSTIC_EXAMPLES
    print("PASS D1 contract tests", flush=True)
    print("  fixed subset=100, Q=20000, seeds=3", flush=True)
    print("  Robust EOT=1; Full EOT=5; MC=1000", flush=True)
    print("  post-hoc only; R5 claim boundary is immutable", flush=True)


if __name__ == "__main__":
    main()
