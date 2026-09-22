#!/usr/bin/env python3
"""Common-MC=1000 re-evaluation of the six saved V7 hybrid candidates.

This stage performs no attack optimization.  It evaluates only the three V7
saved candidates for one defense mode, using the exact Monte-Carlo protocol of
the completed V5/V6 R2 re-evaluation.  The validated V5/V6 R2 rows are reused,
not recomputed, and are combined with the new V7 measurements for paired
V5/V6/V7 comparisons on the same fixed 100 CIFAR-10 images.

The 1000 draws used here are measurement draws.  They are not charged to the
original Q=5000 attack budget and must never be described as attack queries.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Any, Mapping, Sequence

import numpy as np

import reevaluate_topk_v5_v6_candidates_mc1000 as prior


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = Path("./models/resnet20")
DEFAULT_STANDARD_WEIGHTS = MODEL_ROOT / "cifar10/resnet20_cifar10.pth"
DEFAULT_DUAL_WEIGHTS = (
    MODEL_ROOT
    / "fine_frontier_top1_set2/resnet20_staged_epsilon_final_weights.pt"
)
DEFAULT_V7_FULL = MODEL_ROOT / "paper_topk_v7_hybrid_full_k2_q5k_seed3"
DEFAULT_V7_NOISE = (
    MODEL_ROOT / "paper_topk_v7_hybrid_noise_only_k2_q5k_seed3"
)
DEFAULT_PRIOR_FULL = MODEL_ROOT / "paper_topk_candidate_reeval_full_k2_mc1000_r2"
DEFAULT_PRIOR_NOISE = (
    MODEL_ROOT / "paper_topk_candidate_reeval_noise_only_k2_mc1000_r2"
)

V7_SEEDS = (4041, 5051, 6061)
EXPECTED_DATA_SEED = 3031
EXPECTED_K = 2
EXPECTED_EPSILON_PX = 2.0
EXPECTED_BUDGET = 5000
EXPECTED_EXAMPLES = 100
EXPECTED_NATIVE_DRAWS = 100
EXPECTED_COMMON_DRAWS = 1000
EXPECTED_COMMON_SEED = 2026082801
EXPECTED_BOOTSTRAP_RESAMPLES = 50000
EXPECTED_BOOTSTRAP_SEED = 20260827
EXPECTED_SIGMA = 0.02144277648925781
EXPECTED_V7_PROTOCOL = "topk_v7_hybrid_exploration_racing"
EXPECTED_ENGINE_PROTOCOL = "topk_v6_perdraw_square_racing"
EXPECTED_ENGINE_SHA256 = (
    "bfbe2815e60bed15cbeb205b60579f84ead96c63be4c19cde49e0cb71676a8df"
)
EXPECTED_PRIOR_PROTOCOL = "topk_v5_v6_saved_candidate_common_mc1000_r2"
EXPECTED_PRIOR_SCRIPT_SHA256 = (
    "ce7dd89d834c400f48dcbc2c8528772b0c4f4c082d76b4321e0fa49ac744b001"
)
REEVAL_PROTOCOL = "topk_v5_v6_v7_saved_candidate_common_mc1000_r3"
EXPECTED_LAUNCHER_SHA256 = (
    "2cbcb74c15b0b91245d56c7db585f765352c1b598e4908d2b4c2f31cff61bb80"
)
EXPECTED_QUERY_PLAN = {
    "api_queries": 5000,
    "confirmation_draws": [126, 125, 125, 125, 125, 125, 125, 125],
    "confirmation_draws_max": 126,
    "confirmation_draws_min": 125,
    "confirmation_queries": 1001,
    "logical_calls": 3735,
    "race_queries": 408,
    "race_rounds": 17,
    "search_queries": 3999,
    "search_steps": 1795,
}
PRIOR_SOURCE_IDS = (
    "v5_proposed_strongest",
    "v6_proposed_seed_4041",
    "v6_proposed_seed_5051",
    "v6_proposed_seed_6061",
    "v5_robust_anchor",
    "v6_robust_seed_4041",
    "v6_robust_seed_5051",
    "v6_robust_seed_6061",
)
V7_SOURCE_IDS = tuple(f"v7_proposed_seed_{seed}" for seed in V7_SEEDS)
COMBINED_SOURCE_IDS = (
    "v5_proposed_strongest",
    "v6_proposed_seed_4041",
    "v6_proposed_seed_5051",
    "v6_proposed_seed_6061",
    *V7_SOURCE_IDS,
    "v5_robust_anchor",
    "v6_robust_seed_4041",
    "v6_robust_seed_5051",
    "v6_robust_seed_6061",
)
EXPECTED_COMPARISON_COUNT = 12
FLOAT32_AGGREGATE_TOLERANCE_PP = 2e-5


@dataclass(frozen=True)
class V7CandidateSpec:
    source_id: str
    launcher_seed: int
    seed_dir: Path
    config: Mapping[str, Any]
    source_row: Mapping[str, str]
    candidate_path: Path
    candidate_sha256: str

    @property
    def attack_seed(self) -> int:
        return int(self.source_row["attack_seed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate V7 saved candidates under the exact prior common-MC=1000 "
            "protocol, then combine them with validated V5/V6 R2 outputs."
        )
    )
    parser.add_argument("--mode", choices=("full", "noise_only"), required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--prior-r2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-weights", type=Path, default=DEFAULT_STANDARD_WEIGHTS)
    parser.add_argument("--dual-weights", type=Path, default=DEFAULT_DUAL_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--eval-draws", type=int, default=EXPECTED_COMMON_DRAWS)
    parser.add_argument("--reeval-seed", type=int, default=EXPECTED_COMMON_SEED)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=EXPECTED_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=EXPECTED_BOOTSTRAP_SEED)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--rank-temperature", type=float, default=0.5)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(path, csv_text(rows))


def acquire_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".candidate_reeval_v7.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise SystemExit(f"V7 candidate re-evaluation is active (PID {owner}).") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def close(a: float, b: float, tolerance: float = 2e-6) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def validate_manifest(manifest: Mapping[str, Any], *, mode: str) -> None:
    expected_steps = 2 if mode == "full" else 0
    expected = {
        "protocol_version": EXPECTED_V7_PROTOCOL,
        "mode": mode,
        "engine_protocol": EXPECTED_ENGINE_PROTOCOL,
        "engine_attack_name": "SquareRacing",
        "paper_attack_label": "SquareHybrid",
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "launcher_sha256": EXPECTED_LAUNCHER_SHA256,
        "data_seed": EXPECTED_DATA_SEED,
        "k": EXPECTED_K,
        "epsilon_px": EXPECTED_EPSILON_PX,
        "api_budget": EXPECTED_BUDGET,
        "eval_draws": EXPECTED_NATIVE_DRAWS,
        "defenses": ["proposed"],
        "seeds": list(V7_SEEDS),
        "search_eot": 1,
        "refresh_eot": 1,
        "pool_size": 8,
        "race_interval": 100,
        "race_eot": 3,
        "confirm_fraction": 0.2,
        "score_temperature": 0.1,
        "strength_steps": expected_steps,
        "query_plan": EXPECTED_QUERY_PLAN,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V7 manifest mismatch: {mismatches}")


def validate_v7_config(
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    manifest: Mapping[str, Any],
    standard_weights: Path,
    dual_weights: Path,
) -> None:
    prior.validate_config(
        config,
        protocol=EXPECTED_ENGINE_PROTOCOL,
        mode=mode,
        expected_seed=seed,
        standard_weights=standard_weights,
        dual_weights=dual_weights,
    )
    expected = {
        "run_tag": f"v7_hybrid_{mode}_k2_q5000_seed{seed}",
        "defenses": ["proposed"],
        "attacks": ["SquareRacing"],
        "attack_goals": ["exact_set"],
        "budgets": [EXPECTED_BUDGET],
        "budget_unit": "api_queries",
        "eval_draws": EXPECTED_NATIVE_DRAWS,
        "save_candidates": True,
        "candidate_dtype": "float32",
        "racing_search_eot": 1,
        "racing_refresh_eot": 1,
        "racing_pool_size": 8,
        "racing_race_interval": 100,
        "racing_race_eot": 3,
        "racing_confirm_fraction": 0.2,
        "racing_score_temperature": 0.1,
        "evaluator_sha256": EXPECTED_ENGINE_SHA256,
        "reuse_clean_dir": manifest["reused_clean_dir"],
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V7 seed={seed} config mismatch: {mismatches}")
    implementation = config.get("attack_implementation", {})
    if implementation.get("SquareRacing") != (
        "square_attack_linf_per_draw_archive_racing_v6"
    ):
        raise ValueError(f"V7 seed={seed} records the wrong attack implementation")


def validate_native_rows(
    seed_dir: Path,
    source_row: Mapping[str, str],
    *,
    seed: int,
) -> list[dict[str, str]]:
    rows = read_csv(seed_dir / "attack_per_example_topk_v6.csv")
    if len(rows) != EXPECTED_EXAMPLES:
        raise ValueError(f"V7 seed={seed} has {len(rows)} per-example rows")
    rows.sort(key=lambda row: int(row["position"]))
    if [int(row["position"]) for row in rows] != list(range(EXPECTED_EXAMPLES)):
        raise ValueError(f"V7 seed={seed} positions are incomplete or duplicated")
    if len({row["sample_id"] for row in rows}) != EXPECTED_EXAMPLES:
        raise ValueError(f"V7 seed={seed} sample IDs are not unique")
    expected_scalar = {
        "defense": "proposed",
        "attack": "SquareRacing",
        "attack_goal": "exact_set",
        "k": str(EXPECTED_K),
        "epsilon_px": str(EXPECTED_EPSILON_PX),
        "requested_budget": str(EXPECTED_BUDGET),
        "logical_budget": str(EXPECTED_QUERY_PLAN["logical_calls"]),
        "attack_api_queries": str(EXPECTED_BUDGET),
        "eval_draws": str(EXPECTED_NATIVE_DRAWS),
        "candidate_file": source_row["candidate_file"],
    }
    for key, value in expected_scalar.items():
        if any(row.get(key) != value for row in rows):
            raise ValueError(f"V7 seed={seed} per-example mismatch for {key}")
    aggregate = 100.0 * float(
        np.mean([float(row["exact_set_probability"]) for row in rows])
    )
    if not close(
        aggregate,
        float(source_row["exact_set"]),
        tolerance=FLOAT32_AGGREGATE_TOLERANCE_PP,
    ):
        raise ValueError(
            f"V7 seed={seed} native aggregate mismatch: {aggregate} != "
            f"{source_row['exact_set']}"
        )
    return rows


def build_v7_specs(
    *,
    mode: str,
    v7_root: Path,
    standard_weights: Path,
    dual_weights: Path,
) -> tuple[dict[str, Any], list[V7CandidateSpec], dict[str, list[dict[str, str]]]]:
    v7_root = v7_root.expanduser().resolve()
    manifest = read_json(v7_root / "v7_hybrid_manifest.json")
    validate_manifest(manifest, mode=mode)
    specs: list[V7CandidateSpec] = []
    native_rows: dict[str, list[dict[str, str]]] = {}
    for seed in V7_SEEDS:
        seed_dir = v7_root / f"seed_{seed}"
        config = read_json(seed_dir / "run_config_topk_v6.json")
        validate_v7_config(
            config,
            mode=mode,
            seed=seed,
            manifest=manifest,
            standard_weights=standard_weights,
            dual_weights=dual_weights,
        )
        rows = read_csv(seed_dir / "attack_results_topk_v6.csv")
        source_row = prior.select_unique_row(
            rows,
            defense="proposed",
            attack="SquareRacing",
            k=EXPECTED_K,
            budget=EXPECTED_BUDGET,
        )
        expected_row = {
            "eval_draws": str(EXPECTED_NATIVE_DRAWS),
            "examples": str(EXPECTED_EXAMPLES),
            "logical_budget": str(EXPECTED_QUERY_PLAN["logical_calls"]),
            "attack_search_steps": str(EXPECTED_QUERY_PLAN["search_steps"]),
            "attack_search_eot": "1",
            "attack_refresh_eot": "1",
            "attack_confirmation_candidates": "8",
            "attack_confirmation_queries": str(
                EXPECTED_QUERY_PLAN["confirmation_queries"]
            ),
            "attack_confirmation_draws_min": "125",
            "attack_confirmation_draws_max": "126",
            "attack_race_rounds": str(EXPECTED_QUERY_PLAN["race_rounds"]),
            "attack_race_interval": "100",
            "attack_race_eot": "3",
            "attack_race_queries": str(EXPECTED_QUERY_PLAN["race_queries"]),
            "attack_score_temperature": "0.1",
            "attack_score_objective": "mean_per_draw_sigmoid_exact_set_margin",
        }
        mismatches = {
            key: (source_row.get(key), value)
            for key, value in expected_row.items()
            if source_row.get(key) != value
        }
        if mismatches:
            raise ValueError(f"V7 seed={seed} result-row mismatch: {mismatches}")
        candidate_path = prior.resolve_under(seed_dir, source_row["candidate_file"])
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Missing V7 candidate tensor: {candidate_path}")
        source_id = f"v7_proposed_seed_{seed}"
        specs.append(
            V7CandidateSpec(
                source_id=source_id,
                launcher_seed=seed,
                seed_dir=seed_dir,
                config=config,
                source_row=dict(source_row),
                candidate_path=candidate_path,
                candidate_sha256=sha256_file(candidate_path),
            )
        )
        native_rows[source_id] = validate_native_rows(
            seed_dir, source_row, seed=seed
        )
    return manifest, specs, native_rows


def validate_prior_r2(
    directory: Path,
    *,
    mode: str,
    expected_batch_size: int,
    standard_weights: Path,
    dual_weights: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    directory = directory.expanduser().resolve()
    required = (
        "run_config_candidate_reeval.json",
        "candidate_reeval_summary.csv",
        "candidate_reeval_per_example.csv",
        "candidate_reeval_comparisons.csv",
        "candidate_reeval_summary.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Prior R2 directory misses: {', '.join(missing)}")
    config = read_json(directory / "run_config_candidate_reeval.json")
    expected = {
        "protocol_version": EXPECTED_PRIOR_PROTOCOL,
        "mode": mode,
        "batch_size": expected_batch_size,
        "eval_draws": EXPECTED_COMMON_DRAWS,
        "reeval_seed": EXPECTED_COMMON_SEED,
        "bootstrap_resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
        "data_seed": EXPECTED_DATA_SEED,
        "k": EXPECTED_K,
        "epsilon_px": EXPECTED_EPSILON_PX,
        "source_attack_budget": EXPECTED_BUDGET,
        "common_random_numbers": True,
        "reeval_draws_counted_in_attack_budget": False,
        "candidate_source_ids": list(PRIOR_SOURCE_IDS),
        "canonical_clean_source_id": "v6_proposed_seed_4041",
        "script_sha256": EXPECTED_PRIOR_SCRIPT_SHA256,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prior R2 config mismatch: {mismatches}")
    for key, current in (
        ("standard_weights", standard_weights),
        ("dual_weights", dual_weights),
    ):
        configured = Path(str(config.get(key, ""))).expanduser().resolve()
        if configured != current.expanduser().resolve():
            raise ValueError(f"Prior R2 {key} path differs from the current run")
    helper_hash = sha256_file(PROJECT_DIR / prior.__file__)
    if helper_hash != EXPECTED_PRIOR_SCRIPT_SHA256:
        raise ValueError(
            "Bundled R2 helper does not match the script that produced the prior rows: "
            f"{helper_hash}"
        )

    summaries = read_csv(directory / "candidate_reeval_summary.csv")
    examples = read_csv(directory / "candidate_reeval_per_example.csv")
    if len(summaries) != len(PRIOR_SOURCE_IDS) or len(examples) != (
        len(PRIOR_SOURCE_IDS) * EXPECTED_EXAMPLES
    ):
        raise ValueError("Prior R2 summary/per-example grid is incomplete")
    if tuple(row["source_id"] for row in summaries) != PRIOR_SOURCE_IDS:
        raise ValueError("Prior R2 source ordering or IDs differ from the audited grid")
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in examples:
        by_source.setdefault(row["source_id"], []).append(row)
    if set(by_source) != set(PRIOR_SOURCE_IDS):
        raise ValueError("Prior R2 per-example source IDs are incomplete")
    canonical_key: list[tuple[str, str, str, str]] | None = None
    summary_by_id = {row["source_id"]: row for row in summaries}
    for source_id in PRIOR_SOURCE_IDS:
        rows = sorted(by_source[source_id], key=lambda row: int(row["position"]))
        if len(rows) != EXPECTED_EXAMPLES:
            raise ValueError(f"Prior R2 {source_id} has the wrong row count")
        if any(int(row["reeval_seed"]) != EXPECTED_COMMON_SEED for row in rows):
            raise ValueError(f"Prior R2 {source_id} uses the wrong seed")
        summary = summary_by_id[source_id]
        if int(summary["batch_size"]) != expected_batch_size:
            raise ValueError(f"Prior R2 {source_id} uses the wrong batch size")
        expected_draws = EXPECTED_COMMON_DRAWS if "proposed" in source_id else 1
        if int(summary["reeval_draws_actual"]) != expected_draws:
            raise ValueError(f"Prior R2 {source_id} uses the wrong draw count")
        if re.fullmatch(r"[0-9a-f]{64}", summary["source_candidate_sha256"]) is None:
            raise ValueError(f"Prior R2 {source_id} has an invalid candidate hash")
        values = np.array([float(row["exact_set_probability"]) for row in rows])
        aggregate = 100.0 * float(values.mean())
        if not close(
            aggregate,
            float(summary["exact_set"]),
            tolerance=FLOAT32_AGGREGATE_TOLERANCE_PP,
        ):
            raise ValueError(f"Prior R2 {source_id} aggregate does not match its rows")
        key = [
            (
                row["sample_id"],
                row["sample_index"],
                row["true_label"],
                row["reference_set"],
            )
            for row in rows
        ]
        if canonical_key is None:
            canonical_key = key
        elif key != canonical_key:
            raise ValueError(f"Prior R2 sample/reference mismatch for {source_id}")
    return config, summaries, examples


def validate_native_compatibility(
    native_rows: Mapping[str, Sequence[Mapping[str, str]]],
    prior_examples: Sequence[Mapping[str, str]],
) -> None:
    canonical = sorted(
        (row for row in prior_examples if row["source_id"] == "v5_proposed_strongest"),
        key=lambda row: int(row["position"]),
    )
    for source_id, rows in native_rows.items():
        current = sorted(rows, key=lambda row: int(row["position"]))
        for left, right in zip(canonical, current):
            fields = ("position", "sample_id", "sample_index", "true_label", "reference_set")
            if any(left[field] != right[field] for field in fields):
                raise ValueError(
                    f"Native V7 samples/references differ from prior R2 for {source_id}"
                )


def v7_as_prior_spec(spec: V7CandidateSpec) -> prior.CandidateSpec:
    return prior.CandidateSpec(
        source_id=spec.source_id,
        source_version="V7",
        source_protocol=EXPECTED_ENGINE_PROTOCOL,
        source_dir=spec.seed_dir,
        source_row=spec.source_row,
        candidate_path=spec.candidate_path,
    )


def validate_v7_metadata(metadata: Mapping[str, Any], *, spec: V7CandidateSpec) -> None:
    details = metadata.get("attack_details")
    if not isinstance(details, Mapping):
        raise ValueError(f"V7 {spec.source_id} has no attack_details mapping")
    expected = {
        "search_steps": EXPECTED_QUERY_PLAN["search_steps"],
        "search_eot": 1,
        "refresh_eot": 1,
        "race_rounds": EXPECTED_QUERY_PLAN["race_rounds"],
        "race_interval": 100,
        "race_eot": 3,
        "race_queries": EXPECTED_QUERY_PLAN["race_queries"],
        "confirmation_candidates": 8,
        "confirmation_queries": EXPECTED_QUERY_PLAN["confirmation_queries"],
        "confirmation_draws_min": 125,
        "confirmation_draws_max": 126,
        "score_temperature": 0.1,
        "score_objective": "mean_per_draw_sigmoid_exact_set_margin",
        "logical_calls": EXPECTED_QUERY_PLAN["logical_calls"],
        "api_queries": EXPECTED_QUERY_PLAN["api_queries"],
    }
    mismatches = {
        key: (details.get(key), value)
        for key, value in expected.items()
        if details.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V7 {spec.source_id} payload query-plan mismatch: {mismatches}")
    if int(metadata.get("attack_seed", -1)) != spec.attack_seed:
        raise ValueError(f"V7 {spec.source_id} payload attack seed mismatch")
    if str(metadata.get("run_tag")) != str(spec.config["run_tag"]):
        raise ValueError(f"V7 {spec.source_id} payload run tag mismatch")


def validate_payloads(
    torch: Any,
    specs: Sequence[V7CandidateSpec],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = {
        spec.source_id: prior.load_payload(torch, v7_as_prior_spec(spec))
        for spec in specs
    }
    first = specs[0]
    if "clean" not in raw[first.source_id]:
        raise ValueError("The canonical V7 payload does not store its clean tensor")
    canonical = prior.validate_payload(
        torch,
        v7_as_prior_spec(first),
        raw[first.source_id],
        baseline=None,
    )
    baseline = {
        "clean": canonical["clean"],
        "labels": canonical["labels"],
        "dataset_indices": canonical["dataset_indices"],
    }
    checked: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if "clean" not in raw[spec.source_id]:
            raise ValueError(f"V7 {spec.source_id} does not store a clean tensor")
        if spec.source_id == first.source_id:
            payload = canonical
        else:
            payload = prior.validate_payload(
                torch,
                v7_as_prior_spec(spec),
                raw[spec.source_id],
                baseline,
            )
        validate_v7_metadata(payload["metadata"], spec=spec)
        if payload["clean_tensor_source"] != "stored_in_candidate":
            raise ValueError(f"Unexpected clean provenance for {spec.source_id}")
        checked[spec.source_id] = payload
    return checked, baseline


def verify_reference_and_subset(
    torch: Any,
    baseline: Mapping[str, Any],
    reference_logits: Any,
    prior_examples: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    canonical = sorted(
        (row for row in prior_examples if row["source_id"] == "v5_proposed_strongest"),
        key=lambda row: int(row["position"]),
    )
    indices = baseline["dataset_indices"].detach().cpu().tolist()
    labels = baseline["labels"].detach().cpu().tolist()
    reference = torch.topk(reference_logits, EXPECTED_K, dim=1).indices.tolist()
    for position, row in enumerate(canonical):
        if int(row["position"]) != position:
            raise ValueError("Prior R2 positions are not canonical")
        if int(row["sample_index"]) != int(indices[position]):
            raise ValueError("V7 candidate indices differ from prior R2")
        if int(row["true_label"]) != int(labels[position]):
            raise ValueError("V7 candidate labels differ from prior R2")
        if json.loads(row["reference_set"]) != reference[position]:
            raise ValueError(
                "Current standard checkpoint/reference path differs from prior R2"
            )
    return canonical


def build_models(evaluator: Any, torch: Any, args: argparse.Namespace):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        index = device.index if device.index is not None else 0
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)
        torch.cuda.reset_peak_memory_stats(device)
    standard_model, standard_arch = evaluator.legacy.load_resnet20(
        args.standard_weights,
        device,
        evaluator.legacy.STANDARD_CIFAR10_STD,
    )
    dual_model = evaluator.base.load_dual_defense(args.dual_weights, device)
    robust_model = evaluator.base.RobustPixelModel(dual_model).to(device).eval()
    strength_steps = 2 if args.mode == "full" else 0
    proposed_actual = evaluator.TopKTargetedGaussianDefense(
        standard_model=standard_model,
        gradient_model=robust_model,
        output_model=robust_model,
        top_k=EXPECTED_K,
        alpha=0.005,
        epsilon=0.03,
        steps=strength_steps,
        sigma=EXPECTED_SIGMA,
        top1_weight=1.0,
        set_weight=4.0,
    )
    proposed = evaluator.base.DefenseSpec(
        "proposed", proposed_actual, robust_model, EXPECTED_SIGMA
    )
    robust = evaluator.base.DefenseSpec(
        "robust", evaluator.base.make_actual(robust_model), robust_model, 0.0
    )
    return device, standard_model, robust_model, proposed, robust, standard_arch


def validate_robust_anchor(
    *,
    evaluator: Any,
    torch: Any,
    robust_defense: Any,
    reference_logits: Any,
    baseline: Mapping[str, Any],
    prior_config: Mapping[str, Any],
    prior_summaries: Sequence[Mapping[str, str]],
    prior_examples: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = next(
        row for row in prior_summaries if row["source_id"] == "v5_robust_anchor"
    )
    v5_dir = Path(str(prior_config["v5_dir"])).expanduser().resolve()
    result_rows = read_csv(v5_dir / "attack_results_topk_v5.csv")
    source_row = prior.select_unique_row(
        result_rows,
        defense="robust",
        attack="SquareRef",
        k=EXPECTED_K,
        budget=EXPECTED_BUDGET,
    )
    candidate_path = prior.resolve_under(v5_dir, source_row["candidate_file"])
    candidate_hash = sha256_file(candidate_path)
    if candidate_hash != summary["source_candidate_sha256"]:
        raise ValueError("Robust anchor tensor differs from the prior R2 artifact")
    spec = prior.CandidateSpec(
        source_id="v5_robust_anchor",
        source_version="V5",
        source_protocol=prior.EXPECTED_V5_PROTOCOL,
        source_dir=v5_dir,
        source_row=source_row,
        candidate_path=candidate_path,
    )
    raw = prior.load_payload(torch, spec)
    checked = prior.validate_payload(torch, spec, raw, baseline)
    evaluation = prior.evaluate_with_progress(
        evaluator,
        robust_defense,
        checked["adversarial"].to(torch.device(args.device)),
        checked["labels"].to(torch.device(args.device)),
        reference_logits,
        k=EXPECTED_K,
        batch_size=args.batch_size,
        requested_draws=args.eval_draws,
        tau=args.rank_temperature,
        seed=args.reeval_seed,
        source_id="robust_anchor_preflight",
        progress_every_batches=max(1, args.progress_every_batches),
    )
    stored = sorted(
        (row for row in prior_examples if row["source_id"] == "v5_robust_anchor"),
        key=lambda row: int(row["position"]),
    )
    mappings = {
        "top1": "top1_probability",
        "coverage": "coverage_probability",
        "exact_set": "exact_set_probability",
        "reference_top1_match": "reference_top1_match_probability",
        "hidden_topk_failure": "hidden_topk_failure_probability",
    }
    for metric, column in mappings.items():
        observed = evaluation.values[metric].detach().cpu().numpy()
        expected = np.array([float(row[column]) for row in stored])
        if not np.array_equal(observed, expected):
            raise ValueError(f"Robust anchor no longer reproduces metric {metric}")
    reference = evaluation.reference_set.detach().cpu().tolist()
    if reference != [json.loads(row["reference_set"]) for row in stored]:
        raise ValueError("Robust anchor reference sets no longer reproduce")
    return {
        "source_id": "v5_robust_anchor",
        "candidate_sha256": candidate_hash,
        "actual_draws": int(evaluation.draws),
        "examples": EXPECTED_EXAMPLES,
        "discrete_outputs_exactly_reproduced": True,
    }


def partial_paths(output_dir: Path, source_id: str) -> tuple[Path, Path]:
    partials = output_dir / "partials"
    return (
        partials / f"{source_id}.summary.json",
        partials / f"{source_id}.per_example.csv",
    )


def read_completed_partial(
    output_dir: Path,
    spec: V7CandidateSpec,
    *,
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
    summary_path, examples_path = partial_paths(output_dir, spec.source_id)
    if not summary_path.exists() and not examples_path.exists():
        return None
    if not summary_path.is_file() or not examples_path.is_file():
        raise RuntimeError(
            f"Incomplete V7 partial for {spec.source_id}; preserve it for audit"
        )
    summary = read_json(summary_path)
    examples = read_csv(examples_path)
    expected = {
        "source_id": spec.source_id,
        "mode": mode,
        "source_version": "V7",
        "source_protocol": EXPECTED_V7_PROTOCOL,
        "source_engine_protocol": EXPECTED_ENGINE_PROTOCOL,
        "source_launcher_seed": spec.launcher_seed,
        "source_attack_seed": spec.attack_seed,
        "source_candidate_sha256": spec.candidate_sha256,
        "clean_tensor_source": "stored_in_candidate",
        "reeval_seed": args.reeval_seed,
        "reeval_draws_requested": args.eval_draws,
        "reeval_draws_actual": args.eval_draws,
        "batch_size": args.batch_size,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches or len(examples) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"V7 partial provenance mismatch for {spec.source_id}: {mismatches}"
        )
    if any(row.get("source_id") != spec.source_id for row in examples):
        raise RuntimeError(f"V7 partial rows mismatch for {spec.source_id}")
    return summary, examples


def evaluate_one(
    *,
    evaluator: Any,
    torch: Any,
    spec: V7CandidateSpec,
    payload: Mapping[str, Any],
    defense: Any,
    reference_logits: Any,
    canonical_prior_rows: Sequence[Mapping[str, str]],
    mode: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    evaluation = prior.evaluate_with_progress(
        evaluator,
        defense,
        payload["adversarial"].to(torch.device(args.device)),
        payload["labels"].to(torch.device(args.device)),
        reference_logits,
        k=EXPECTED_K,
        batch_size=args.batch_size,
        requested_draws=args.eval_draws,
        tau=args.rank_temperature,
        seed=args.reeval_seed,
        source_id=spec.source_id,
        progress_every_batches=args.progress_every_batches,
    )
    wall_seconds = time.perf_counter() - started
    exact_values = evaluation.values["exact_set"].detach().cpu().numpy()
    mc_se = 100.0 * float(
        np.sqrt(np.sum(exact_values * (1.0 - exact_values)))
        / (len(exact_values) * np.sqrt(max(1, evaluation.draws)))
    )
    summary: dict[str, Any] = {
        "source_id": spec.source_id,
        "mode": mode,
        "source_version": "V7",
        "source_protocol": EXPECTED_V7_PROTOCOL,
        "source_engine_protocol": EXPECTED_ENGINE_PROTOCOL,
        "source_paper_attack": "SquareHybrid",
        "source_attack": "SquareRacing",
        "source_defense": "proposed",
        "source_launcher_seed": spec.launcher_seed,
        "source_attack_seed": spec.attack_seed,
        "source_reeval_seed": int(spec.source_row["reeval_seed"]),
        "source_eval_draws": int(spec.source_row["eval_draws"]),
        "source_reported_exact_set": float(spec.source_row["exact_set"]),
        "source_candidate_file": str(spec.source_row["candidate_file"]),
        "source_candidate_sha256": spec.candidate_sha256,
        "candidate_linf_px": payload["linf_px"],
        "clean_tensor_source": payload["clean_tensor_source"],
        "measurement_origin": "new_v7_common_mc1000",
        "data_seed": EXPECTED_DATA_SEED,
        "k": EXPECTED_K,
        "epsilon_px": EXPECTED_EPSILON_PX,
        "original_attack_api_queries": EXPECTED_BUDGET,
        "reeval_seed": args.reeval_seed,
        "reeval_draws_requested": args.eval_draws,
        "reeval_draws_actual": evaluation.draws,
        "reeval_draws_counted_in_attack_budget": False,
        "common_random_numbers": True,
        "examples": EXPECTED_EXAMPLES,
        "batch_size": args.batch_size,
        "wall_seconds": wall_seconds,
        "mc_only_standard_error_pp": mc_se,
    }
    summary.update(evaluation.summary)
    for name, value in evaluation.mean_logits_summary.items():
        summary[f"mean_logits_{name}"] = value
    rows = evaluator.per_example_rows(
        evaluation=evaluation,
        labels=payload["labels"],
        dataset_indices=payload["dataset_indices"],
        defense="proposed",
        condition="saved_candidate_reeval",
        k=EXPECTED_K,
        data_seed=EXPECTED_DATA_SEED,
        attack_seed=spec.attack_seed,
        reeval_seed=args.reeval_seed,
        run_tag=f"common_mc{args.eval_draws}_v7_{mode}",
        shard_count=1,
        shard_index=0,
        attack="SquareRacing",
        attack_goal="exact_set",
        epsilon_px=EXPECTED_EPSILON_PX,
        requested_budget=EXPECTED_BUDGET,
        logical_budget="source_attack",
        attack_eot="source_attack",
        attack_api_queries=EXPECTED_BUDGET,
        candidate_file=str(spec.source_row["candidate_file"]),
    )
    enriched = [
        {
            "source_id": spec.source_id,
            "mode": mode,
            "source_version": "V7",
            "source_protocol": EXPECTED_V7_PROTOCOL,
            "source_engine_protocol": EXPECTED_ENGINE_PROTOCOL,
            "source_paper_attack": "SquareHybrid",
            "source_attack": "SquareRacing",
            "source_defense": "proposed",
            "source_launcher_seed": spec.launcher_seed,
            "source_attack_seed": spec.attack_seed,
            "source_candidate_sha256": spec.candidate_sha256,
            "measurement_origin": "new_v7_common_mc1000",
            "common_random_numbers": True,
            "reeval_draws_counted_in_attack_budget": False,
            **row,
        }
        for row in rows
    ]
    current = sorted(enriched, key=lambda row: int(row["position"]))
    for left, right in zip(canonical_prior_rows, current):
        fields = ("position", "sample_id", "sample_index", "true_label", "reference_set")
        if any(str(left[field]) != str(right[field]) for field in fields):
            raise RuntimeError(
                f"Common-MC output alignment failed for {spec.source_id}"
            )
    summary_path, examples_path = partial_paths(output_dir, spec.source_id)
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))
    atomic_write_csv(examples_path, enriched)
    return summary, enriched


def infer_launcher_seed(source_id: str) -> int | str:
    match = re.search(r"seed_(\d+)$", source_id)
    if match:
        return int(match.group(1))
    return 4041


def enrich_prior_summary(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        **row,
        "source_engine_protocol": row["source_protocol"],
        "source_paper_attack": row["source_attack"],
        "source_launcher_seed": infer_launcher_seed(row["source_id"]),
        "measurement_origin": "reused_prior_r2",
    }


def enrich_prior_example(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        **row,
        "source_engine_protocol": row["source_protocol"],
        "source_paper_attack": row["source_attack"],
        "source_launcher_seed": infer_launcher_seed(row["source_id"]),
        "measurement_origin": "reused_prior_r2",
    }


def classify_generic(low: float, high: float) -> str:
    if low > 0:
        return "baseline_attack_stronger"
    if high < 0:
        return "candidate_attack_stronger"
    return "statistically_indistinguishable"


def build_comparisons(
    summaries: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    summary_by_id = {str(row["source_id"]): row for row in summaries}
    by_source: dict[str, dict[str, float]] = {}
    for row in examples:
        if row["source_defense"] != "proposed":
            continue
        by_source.setdefault(str(row["source_id"]), {})[str(row["sample_id"])] = float(
            row["exact_set_probability"]
        )
    canonical_ids = sorted(by_source["v5_proposed_strongest"])
    arrays = {
        source_id: np.array([values[item] for item in canonical_ids], dtype=np.float64)
        for source_id, values in by_source.items()
    }
    for source_id, values in by_source.items():
        if sorted(values) != canonical_ids:
            raise RuntimeError(f"Sample IDs differ for {source_id}")

    rows: list[dict[str, Any]] = []

    def append_comparison(
        *,
        name: str,
        family: str,
        baseline_id: str,
        candidate_id: str,
        baseline: np.ndarray,
        candidate: np.ndarray,
        seed_label: int | str,
        bootstrap_offset: int,
        inference_scope: str,
        baseline_seed_count: int,
        candidate_seed_count: int,
    ) -> None:
        mean, low, high = prior.paired_bootstrap(
            candidate - baseline,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + bootstrap_offset,
        )
        verdict = classify_generic(low, high)
        if verdict == "baseline_attack_stronger":
            verdict_label = f"{baseline_id}_stronger"
        elif verdict == "candidate_attack_stronger":
            verdict_label = f"{candidate_id}_stronger"
        else:
            verdict_label = verdict
        rows.append(
            {
                "comparison": name,
                "comparison_family": family,
                "baseline_source_id": baseline_id,
                "candidate_source_id": candidate_id,
                "attack_seed": seed_label,
                "baseline_exact_set_pct": 100.0 * float(baseline.mean()),
                "candidate_exact_set_pct": 100.0 * float(candidate.mean()),
                "candidate_minus_baseline_pp": 100.0 * mean,
                "ci95_low_pp": 100.0 * low,
                "ci95_high_pp": 100.0 * high,
                "positive_delta_means": (
                    "candidate attack is weaker because it leaves higher survival"
                ),
                "verdict": verdict,
                "verdict_label": verdict_label,
                "inference_scope": inference_scope,
                "baseline_seed_count": baseline_seed_count,
                "candidate_seed_count": candidate_seed_count,
                "bootstrap_resamples": bootstrap_resamples,
                "bootstrap_seed": bootstrap_seed + bootstrap_offset,
            }
        )

    v5 = arrays["v5_proposed_strongest"]
    v6_arrays = [arrays[f"v6_proposed_seed_{seed}"] for seed in V7_SEEDS]
    v7_arrays = [arrays[f"v7_proposed_seed_{seed}"] for seed in V7_SEEDS]
    fixed_scope = (
        "paired bootstrap over the fixed 100 images; attack seeds are observed runs"
    )
    for offset, seed in enumerate(V7_SEEDS):
        append_comparison(
            name=f"v6_proposed_seed_{seed}_minus_v5_proposed_strongest",
            family="V6_vs_V5",
            baseline_id="v5_proposed_strongest",
            candidate_id=f"v6_proposed_seed_{seed}",
            baseline=v5,
            candidate=v6_arrays[offset],
            seed_label=seed,
            bootstrap_offset=offset,
            inference_scope=fixed_scope,
            baseline_seed_count=1,
            candidate_seed_count=1,
        )
    v6_mean = np.stack(v6_arrays).mean(axis=0)
    append_comparison(
        name="v6_three_seed_mean_minus_v5_proposed_strongest",
        family="V6_vs_V5",
        baseline_id="v5_proposed_strongest",
        candidate_id="v6_three_seed_mean",
        baseline=v5,
        candidate=v6_mean,
        seed_label="4041,5051,6061",
        bootstrap_offset=100,
        inference_scope=(
            "paired bootstrap over the fixed 100 images after averaging three observed "
            "V6 seeds; V5 has one observed attack seed"
        ),
        baseline_seed_count=1,
        candidate_seed_count=3,
    )

    for offset, seed in enumerate(V7_SEEDS):
        append_comparison(
            name=f"v7_proposed_seed_{seed}_minus_v5_proposed_strongest",
            family="V7_vs_V5",
            baseline_id="v5_proposed_strongest",
            candidate_id=f"v7_proposed_seed_{seed}",
            baseline=v5,
            candidate=v7_arrays[offset],
            seed_label=seed,
            bootstrap_offset=200 + offset,
            inference_scope=fixed_scope,
            baseline_seed_count=1,
            candidate_seed_count=1,
        )
    v7_mean = np.stack(v7_arrays).mean(axis=0)
    append_comparison(
        name="v7_three_seed_mean_minus_v5_proposed_strongest",
        family="V7_vs_V5",
        baseline_id="v5_proposed_strongest",
        candidate_id="v7_three_seed_mean",
        baseline=v5,
        candidate=v7_mean,
        seed_label="4041,5051,6061",
        bootstrap_offset=300,
        inference_scope=(
            "paired bootstrap over the fixed 100 images after averaging three observed "
            "V7 seeds; V5 has one observed attack seed"
        ),
        baseline_seed_count=1,
        candidate_seed_count=3,
    )

    for offset, seed in enumerate(V7_SEEDS):
        append_comparison(
            name=f"v7_proposed_seed_{seed}_minus_v6_proposed_seed_{seed}",
            family="V7_vs_V6_matched_seed",
            baseline_id=f"v6_proposed_seed_{seed}",
            candidate_id=f"v7_proposed_seed_{seed}",
            baseline=v6_arrays[offset],
            candidate=v7_arrays[offset],
            seed_label=seed,
            bootstrap_offset=400 + offset,
            inference_scope=(
                "paired bootstrap over the fixed 100 images for matched observed "
                "launcher seeds; it does not generalize over attack seeds"
            ),
            baseline_seed_count=1,
            candidate_seed_count=1,
        )
    append_comparison(
        name="v7_three_seed_mean_minus_v6_three_seed_mean",
        family="V7_vs_V6_mean",
        baseline_id="v6_three_seed_mean",
        candidate_id="v7_three_seed_mean",
        baseline=v6_mean,
        candidate=v7_mean,
        seed_label="4041,5051,6061",
        bootstrap_offset=500,
        inference_scope=(
            "paired bootstrap over the fixed 100 images after averaging the same three "
            "observed launcher seeds for V6 and V7; attack-seed uncertainty is not "
            "included in this interval"
        ),
        baseline_seed_count=3,
        candidate_seed_count=3,
    )
    if len(rows) != EXPECTED_COMPARISON_COUNT:
        raise RuntimeError("Internal comparison-grid error")
    return rows


def attack_envelope(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposed = [row for row in summaries if str(row["source_defense"]) == "proposed"]
    by_version: dict[str, list[Mapping[str, Any]]] = {}
    for row in proposed:
        by_version.setdefault(str(row["source_version"]), []).append(row)
    versions: dict[str, Any] = {}
    for version, rows in by_version.items():
        values = [float(row["exact_set"]) for row in rows]
        strongest = min(rows, key=lambda row: float(row["exact_set"]))
        versions[version] = {
            "observed_seed_count": len(rows),
            "mean_exact_set_pct": float(np.mean(values)),
            "seed_sd_pp": float(stdev(values)) if len(values) > 1 else None,
            "strongest_observed_source_id": strongest["source_id"],
            "strongest_observed_exact_set_pct": float(strongest["exact_set"]),
        }
    strongest_overall = min(proposed, key=lambda row: float(row["exact_set"]))
    return {
        "by_version": versions,
        "overall_strongest_observed_source_id": strongest_overall["source_id"],
        "overall_attack_envelope_exact_set_pct": float(strongest_overall["exact_set"]),
        "direction_rule": "lower Exact-set survival means a stronger attack",
        "selection_caveat": (
            "This minimum is a conservative descriptive attack envelope. Confidence "
            "intervals for pre-specified pairwise comparisons must not be presented as "
            "post-selection uncertainty for the minimum."
        ),
    }


def normalize_config(
    *,
    args: argparse.Namespace,
    specs: Sequence[V7CandidateSpec],
    prior_config: Mapping[str, Any],
    anchor_validation: Mapping[str, Any],
) -> dict[str, Any]:
    prior_dir = args.prior_r2_dir.expanduser().resolve()
    evaluator_path = PROJECT_DIR / "evaluate_general_topk_defense_v6_cifar10.py"
    helper_path = PROJECT_DIR / "reevaluate_topk_v5_v6_candidates_mc1000.py"
    return {
        "protocol_version": REEVAL_PROTOCOL,
        "mode": args.mode,
        "v7_root": str(args.v7_root.expanduser().resolve()),
        "prior_r2_dir": str(prior_dir),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "standard_weights": str(args.standard_weights.expanduser().resolve()),
        "dual_weights": str(args.dual_weights.expanduser().resolve()),
        "standard_weights_sha256": sha256_file(args.standard_weights.expanduser().resolve()),
        "dual_weights_sha256": sha256_file(args.dual_weights.expanduser().resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "eval_draws": args.eval_draws,
        "reeval_seed": args.reeval_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "rank_temperature": args.rank_temperature,
        "data_seed": EXPECTED_DATA_SEED,
        "k": EXPECTED_K,
        "epsilon_px": EXPECTED_EPSILON_PX,
        "source_attack_budget": EXPECTED_BUDGET,
        "candidate_source_ids": list(COMBINED_SOURCE_IDS),
        "reused_source_ids": list(PRIOR_SOURCE_IDS),
        "newly_evaluated_source_ids": [spec.source_id for spec in specs],
        "new_candidate_sha256": {
            spec.source_id: spec.candidate_sha256 for spec in specs
        },
        "common_random_numbers": True,
        "reeval_draws_counted_in_attack_budget": False,
        "reuse_justification": (
            "Prior V5/V6 rows use the same model paths, evaluator, sample order, "
            "batch size, MC draw count, and re-evaluation seed. A deterministic robust "
            "anchor is re-run before V7 measurement."
        ),
        "prior_protocol_version": prior_config["protocol_version"],
        "prior_run_config_sha256": sha256_file(
            prior_dir / "run_config_candidate_reeval.json"
        ),
        "prior_summary_csv_sha256": sha256_file(
            prior_dir / "candidate_reeval_summary.csv"
        ),
        "prior_per_example_csv_sha256": sha256_file(
            prior_dir / "candidate_reeval_per_example.csv"
        ),
        "prior_helper_sha256": sha256_file(helper_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "robust_anchor_validation": dict(anchor_validation),
    }


def validate_args(args: argparse.Namespace) -> None:
    expected_batch = 5 if args.mode == "full" else 25
    if args.batch_size != expected_batch:
        raise ValueError(
            f"{args.mode} must use batch_size={expected_batch} to reproduce prior RNG "
            "layout; received {args.batch_size}"
        )
    if args.eval_draws != EXPECTED_COMMON_DRAWS:
        raise ValueError("This protocol is fixed at MC=1000")
    if args.reeval_seed != EXPECTED_COMMON_SEED:
        raise ValueError("This protocol must use the prior common seed 2026082801")
    if args.bootstrap_resamples != EXPECTED_BOOTSTRAP_RESAMPLES:
        raise ValueError("This protocol is fixed at 50,000 bootstrap resamples")
    if args.bootstrap_seed != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError("This protocol must use bootstrap seed 20260827")
    if args.progress_every_batches < 1:
        raise ValueError("progress_every_batches must be positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0,1]")
    required = [
        args.standard_weights,
        args.dual_weights,
        args.v7_root / "v7_hybrid_manifest.json",
        args.prior_r2_dir / "run_config_candidate_reeval.json",
        PROJECT_DIR / "evaluate_general_topk_defense_v6_cifar10.py",
        PROJECT_DIR / "reevaluate_topk_v5_v6_candidates_mc1000.py",
        PROJECT_DIR / "evaluate_final_top1_set2_defense_cifar10.py",
        PROJECT_DIR / "evaluate_top1_set2_gated_cifar10.py",
    ]
    for seed in V7_SEEDS:
        required.extend(
            [
                args.v7_root / f"seed_{seed}/run_config_topk_v6.json",
                args.v7_root / f"seed_{seed}/attack_results_topk_v6.csv",
                args.v7_root / f"seed_{seed}/attack_per_example_topk_v6.csv",
            ]
        )
    missing = [str(path) for path in required if not path.expanduser().is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    lock = acquire_lock(output_dir)
    try:
        prior_config, prior_summaries, prior_examples = validate_prior_r2(
            args.prior_r2_dir,
            mode=args.mode,
            expected_batch_size=args.batch_size,
            standard_weights=args.standard_weights,
            dual_weights=args.dual_weights,
        )
        manifest, specs, native_rows = build_v7_specs(
            mode=args.mode,
            v7_root=args.v7_root,
            standard_weights=args.standard_weights,
            dual_weights=args.dual_weights,
        )
        validate_native_compatibility(native_rows, prior_examples)
        evaluator_hash = sha256_file(
            PROJECT_DIR / "evaluate_general_topk_defense_v6_cifar10.py"
        )
        if evaluator_hash != manifest["engine_sha256"]:
            raise ValueError(
                f"Current evaluator hash {evaluator_hash} differs from V7 source runs"
            )

        # Lazy imports keep the structural self-test usable outside the GPU server.
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        sys.path.insert(0, str(PROJECT_DIR))
        import evaluate_general_topk_defense_v6_cifar10 as evaluator

        payloads, baseline = validate_payloads(torch, specs)
        (
            device,
            standard_model,
            _robust_model,
            proposed_defense,
            robust_defense,
            standard_arch,
        ) = build_models(evaluator, torch, args)
        standard_spec = evaluator.base.DefenseSpec(
            "standard",
            evaluator.base.make_actual(standard_model),
            standard_model,
            0.0,
        )
        clean_device = baseline["clean"].to(device)
        reference_logits = evaluator.base.compute_logits_batched(
            standard_spec, clean_device, args.batch_size, 1
        ).detach().cpu()
        canonical_prior_rows = verify_reference_and_subset(
            torch, baseline, reference_logits, prior_examples
        )
        anchor_validation = validate_robust_anchor(
            evaluator=evaluator,
            torch=torch,
            robust_defense=robust_defense,
            reference_logits=reference_logits,
            baseline=baseline,
            prior_config=prior_config,
            prior_summaries=prior_summaries,
            prior_examples=prior_examples,
            args=args,
        )
        print(f"PASS robust anchor: {anchor_validation}", flush=True)

        config = normalize_config(
            args=args,
            specs=specs,
            prior_config=prior_config,
            anchor_validation=anchor_validation,
        )
        config_path = output_dir / "run_config_candidate_reeval_v7.json"
        if config_path.is_file():
            existing = read_json(config_path)
            if existing != config:
                raise RuntimeError(
                    "Output directory contains a different V7 re-evaluation config; "
                    "use a fresh output directory."
                )
        else:
            atomic_write_text(config_path, json.dumps(config, indent=2, sort_keys=True))

        print(f"Standard checkpoint architecture: {standard_arch}", flush=True)
        print(
            f"V7 common re-evaluation: mode={args.mode}, MC={args.eval_draws}, "
            f"seed={args.reeval_seed}, new_candidates={len(specs)}",
            flush=True,
        )
        new_summaries: list[dict[str, Any]] = []
        new_examples: list[dict[str, Any]] = []
        for position, spec in enumerate(specs, start=1):
            completed = read_completed_partial(
                output_dir, spec, mode=args.mode, args=args
            )
            if completed is not None:
                summary, examples = completed
                print(f"PASS reusing completed V7 partial {spec.source_id}", flush=True)
            else:
                print(
                    f"[{position}/{len(specs)}] Evaluating {spec.source_id} "
                    f"with common MC={args.eval_draws}",
                    flush=True,
                )
                summary, examples = evaluate_one(
                    evaluator=evaluator,
                    torch=torch,
                    spec=spec,
                    payload=payloads[spec.source_id],
                    defense=proposed_defense,
                    reference_logits=reference_logits,
                    canonical_prior_rows=canonical_prior_rows,
                    mode=args.mode,
                    output_dir=output_dir,
                    args=args,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"PASS {spec.source_id}: Exact-set={summary['exact_set']:.4f}%",
                    flush=True,
                )
            new_summaries.append(summary)
            new_examples.extend(examples)

        summary_by_id: dict[str, Mapping[str, Any]] = {
            row["source_id"]: enrich_prior_summary(row) for row in prior_summaries
        }
        summary_by_id.update({row["source_id"]: row for row in new_summaries})
        combined_summaries = [dict(summary_by_id[source_id]) for source_id in COMBINED_SOURCE_IDS]

        examples_by_id: dict[str, list[dict[str, Any]]] = {}
        for row in prior_examples:
            examples_by_id.setdefault(row["source_id"], []).append(
                enrich_prior_example(row)
            )
        for row in new_examples:
            examples_by_id.setdefault(str(row["source_id"]), []).append(dict(row))
        combined_examples: list[dict[str, Any]] = []
        for source_id in COMBINED_SOURCE_IDS:
            rows = sorted(examples_by_id[source_id], key=lambda row: int(row["position"]))
            if len(rows) != EXPECTED_EXAMPLES:
                raise RuntimeError(f"Combined row count mismatch for {source_id}")
            combined_examples.extend(rows)

        comparisons = build_comparisons(
            combined_summaries,
            combined_examples,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        envelope = attack_envelope(combined_summaries)
        atomic_write_csv(
            output_dir / "candidate_reeval_v5_v6_v7_summary.csv",
            combined_summaries,
        )
        atomic_write_csv(
            output_dir / "candidate_reeval_v5_v6_v7_per_example.csv",
            combined_examples,
        )
        atomic_write_csv(
            output_dir / "candidate_reeval_v5_v6_v7_comparisons.csv",
            comparisons,
        )
        final_summary = {
            "protocol_version": REEVAL_PROTOCOL,
            "mode": args.mode,
            "candidate_count": len(combined_summaries),
            "new_candidate_count": len(new_summaries),
            "per_example_rows": len(combined_examples),
            "comparison_count": len(comparisons),
            "common_reeval_seed": args.reeval_seed,
            "requested_mc_draws": args.eval_draws,
            "robust_anchor_validation": anchor_validation,
            "attack_envelope": envelope,
            "source_exact_set_pct": {
                row["source_id"]: float(row["exact_set"])
                for row in combined_summaries
            },
            "source_candidate_sha256": {
                row["source_id"]: row["source_candidate_sha256"]
                for row in combined_summaries
            },
            "interpretation_rule": (
                "Lower Exact-set survival means a stronger attack. Positive "
                "candidate-minus-baseline deltas mean the candidate attack is weaker."
            ),
            "measurement_scope": (
                "All intervals are paired over the fixed 100 images. Attack seeds are "
                "observed runs, not a sampled population. MC=1000 draws are measurement "
                "draws outside the Q=5000 attack budget."
            ),
        }
        atomic_write_text(
            output_dir / "candidate_reeval_v5_v6_v7_summary.json",
            json.dumps(final_summary, indent=2, sort_keys=True),
        )

        print("\nCommon-MC V5/V6/V7 comparisons", flush=True)
        for row in comparisons:
            print(
                f"{row['comparison']}: "
                f"delta={float(row['candidate_minus_baseline_pp']):+.4f} pp "
                f"[{float(row['ci95_low_pp']):+.4f}, "
                f"{float(row['ci95_high_pp']):+.4f}] -> {row['verdict']}",
                flush=True,
            )
        print(
            "Strongest observed attack envelope: "
            f"{envelope['overall_strongest_observed_source_id']} = "
            f"{envelope['overall_attack_envelope_exact_set_pct']:.4f}%",
            flush=True,
        )
        print(f"V7 candidate re-evaluation complete: {output_dir}", flush=True)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
