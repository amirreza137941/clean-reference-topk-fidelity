#!/usr/bin/env python3
"""CPU-only structural tests for the V7 common-MC=1000 diagnostic."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reevaluate_topk_v7_candidates_mc1000 import (
    COMBINED_SOURCE_IDS,
    EXPECTED_ENGINE_SHA256,
    EXPECTED_PRIOR_SCRIPT_SHA256,
    EXPECTED_QUERY_PLAN,
    PRIOR_SOURCE_IDS,
    V7_SEEDS,
    V7_SOURCE_IDS,
    attack_envelope,
    build_comparisons,
    classify_generic,
    sha256_file,
    validate_manifest,
)
from topk_v7_candidate_reeval_launcher_common import (
    OUTPUT_FULL,
    OUTPUT_NOISE,
    PRIOR_FULL,
    PRIOR_NOISE,
    V7_FULL,
    V7_NOISE,
    reeval_args,
)


PROJECT_DIR = Path(__file__).resolve().parent


def manifest_fixture(mode: str) -> dict:
    return {
        "api_budget": 5000,
        "confirm_fraction": 0.2,
        "data_seed": 3031,
        "defenses": ["proposed"],
        "engine_attack_name": "SquareRacing",
        "engine_protocol": "topk_v6_perdraw_square_racing",
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "epsilon_px": 2.0,
        "eval_draws": 100,
        "k": 2,
        "launcher_sha256": (
            "2cbcb74c15b0b91245d56c7db585f765352c1b598e4908d2b4c2f31cff61bb80"
        ),
        "mode": mode,
        "paper_attack_label": "SquareHybrid",
        "pool_size": 8,
        "protocol_version": "topk_v7_hybrid_exploration_racing",
        "query_plan": copy.deepcopy(EXPECTED_QUERY_PLAN),
        "race_eot": 3,
        "race_interval": 100,
        "refresh_eot": 1,
        "score_temperature": 0.1,
        "search_eot": 1,
        "seeds": list(V7_SEEDS),
        "strength_steps": 2 if mode == "full" else 0,
    }


def comparison_fixture():
    summaries = []
    examples = []
    probabilities = {
        "v5_proposed_strongest": 0.40,
        "v6_proposed_seed_4041": 0.45,
        "v6_proposed_seed_5051": 0.44,
        "v6_proposed_seed_6061": 0.46,
        "v7_proposed_seed_4041": 0.35,
        "v7_proposed_seed_5051": 0.36,
        "v7_proposed_seed_6061": 0.34,
    }
    for source_id, probability in probabilities.items():
        version = source_id[0:2].upper()
        summaries.append(
            {
                "source_id": source_id,
                "source_version": version,
                "source_defense": "proposed",
                "exact_set": 100.0 * probability,
            }
        )
        for position in range(100):
            examples.append(
                {
                    "source_id": source_id,
                    "source_defense": "proposed",
                    "sample_id": f"sample-{position}",
                    "exact_set_probability": probability,
                }
            )
    return summaries, examples


class V7CommonMCTests(unittest.TestCase):
    def test_full_manifest_is_accepted(self) -> None:
        validate_manifest(manifest_fixture("full"), mode="full")

    def test_noise_manifest_is_accepted(self) -> None:
        validate_manifest(manifest_fixture("noise_only"), mode="noise_only")

    def test_query_plan_tamper_is_rejected(self) -> None:
        manifest = manifest_fixture("full")
        manifest["query_plan"]["confirmation_queries"] = 1000
        with self.assertRaises(ValueError):
            validate_manifest(manifest, mode="full")

    def test_verdict_direction_is_explicit(self) -> None:
        self.assertEqual(classify_generic(0.01, 0.03), "baseline_attack_stronger")
        self.assertEqual(classify_generic(-0.03, -0.01), "candidate_attack_stronger")
        self.assertEqual(
            classify_generic(-0.01, 0.01), "statistically_indistinguishable"
        )

    def test_comparison_grid_has_twelve_rows(self) -> None:
        summaries, examples = comparison_fixture()
        rows = build_comparisons(
            summaries,
            examples,
            bootstrap_resamples=5000,
            bootstrap_seed=9,
        )
        self.assertEqual(len(rows), 12)
        mean_row = next(
            row
            for row in rows
            if row["comparison"] == "v7_three_seed_mean_minus_v6_three_seed_mean"
        )
        self.assertAlmostEqual(mean_row["candidate_minus_baseline_pp"], -10.0)
        self.assertEqual(mean_row["verdict"], "candidate_attack_stronger")

    def test_attack_envelope_uses_lower_survival(self) -> None:
        summaries, _ = comparison_fixture()
        envelope = attack_envelope(summaries)
        self.assertEqual(
            envelope["overall_strongest_observed_source_id"],
            "v7_proposed_seed_6061",
        )
        self.assertAlmostEqual(envelope["overall_attack_envelope_exact_set_pct"], 34.0)

    def test_full_launcher_reproduces_prior_rng_layout(self) -> None:
        args = reeval_args(
            mode="full",
            v7_root=V7_FULL,
            prior_r2_dir=PRIOR_FULL,
            output_dir=OUTPUT_FULL,
        )
        joined = " ".join(args)
        self.assertIn("--eval-draws 1000", joined)
        self.assertIn("--reeval-seed 2026082801", joined)
        self.assertIn("--batch-size 5", joined)
        self.assertTrue(str(OUTPUT_FULL).endswith("_full_k2_mc1000_r3"))

    def test_noise_launcher_reproduces_prior_rng_layout(self) -> None:
        args = reeval_args(
            mode="noise_only",
            v7_root=V7_NOISE,
            prior_r2_dir=PRIOR_NOISE,
            output_dir=OUTPUT_NOISE,
        )
        joined = " ".join(args)
        self.assertIn("--eval-draws 1000", joined)
        self.assertIn("--reeval-seed 2026082801", joined)
        self.assertIn("--batch-size 25", joined)
        self.assertTrue(str(OUTPUT_NOISE).endswith("_noise_only_k2_mc1000_r3"))

    def test_source_grid_keeps_prior_rows_and_adds_only_v7(self) -> None:
        self.assertEqual(len(PRIOR_SOURCE_IDS), 8)
        self.assertEqual(len(V7_SOURCE_IDS), 3)
        self.assertEqual(len(COMBINED_SOURCE_IDS), 11)
        self.assertEqual(set(COMBINED_SOURCE_IDS) - set(PRIOR_SOURCE_IDS), set(V7_SOURCE_IDS))

    def test_bundled_provenance_hashes_are_exact(self) -> None:
        self.assertEqual(
            sha256_file(PROJECT_DIR / "reevaluate_topk_v5_v6_candidates_mc1000.py"),
            EXPECTED_PRIOR_SCRIPT_SHA256,
        )
        self.assertEqual(
            sha256_file(PROJECT_DIR / "evaluate_general_topk_defense_v6_cifar10.py"),
            EXPECTED_ENGINE_SHA256,
        )

    def test_output_path_escape_helper_remains_safe(self) -> None:
        # The R2 helper used by this stage must continue to reject candidate paths
        # that escape a source result directory.
        import reevaluate_topk_v5_v6_candidates_mc1000 as prior

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                prior.resolve_under(Path(directory), "../outside.pt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
