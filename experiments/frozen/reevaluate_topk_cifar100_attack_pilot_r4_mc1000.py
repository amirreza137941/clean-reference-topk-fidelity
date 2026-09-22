#!/usr/bin/env python3
"""Common-random-number MC=1000 reevaluation and pilot attack selection."""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import calibrate_topk_cifar100_actual_defense_r1 as r1
import topk_cifar100_attack_pilot_r4_common as common
from search_topk_cifar100_attack_pilot_r4 import (
    FrozenDefense,
    atomic_torch_save,
    load_models,
    payload_identity,
    validate_payload,
)


METRICS = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "exact_set_probability",
    "label_coverage_probability",
    "mean_logits_top1_accuracy",
    "mean_logits_exact_set",
    "mean_boundary_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-100 R4 common MC=1000 reevaluation")
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=common.REEVAL_BATCH_SIZE)
    parser.add_argument("--mc-chunk", type=int, default=common.REEVAL_MC_CHUNK)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def autocast_off(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


def _evaluation_hash_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"values", "evaluation_hash"}
    result = {key: value for key, value in payload.items() if key not in excluded}
    values = payload["values"]
    result["value_sha256"] = {
        key: common.tensor_sha256(values[key]) for key in sorted(values)
    }
    return result


def add_evaluation_hash(payload: dict[str, Any]) -> None:
    payload["value_sha256"] = {
        key: common.tensor_sha256(payload["values"][key])
        for key in sorted(payload["values"])
    }
    payload["evaluation_hash"] = common.object_sha256(_evaluation_hash_input(payload))


def validate_evaluation_payload(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Evaluation cache identity mismatch: {key}")
    values = payload.get("values", {})
    if set(values) != set(METRICS):
        raise AssertionError("Evaluation cache metric set mismatch")
    examples = int(payload.get("examples", -1))
    for key in METRICS:
        tensor = values[key].detach().cpu()
        if tensor.shape != (examples,) or not torch.isfinite(tensor).all():
            raise AssertionError(f"Evaluation cache tensor invalid: {key}")
        if common.tensor_sha256(tensor) != payload.get("value_sha256", {}).get(key):
            raise AssertionError(f"Evaluation cache tensor hash mismatch: {key}")
    if common.object_sha256(_evaluation_hash_input(payload)) != payload.get("evaluation_hash"):
        raise AssertionError("Evaluation cache payload hash mismatch")


def evaluate_images(
    *,
    defense: FrozenDefense,
    images_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    reference_top2_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    mc_chunk: int,
    draws: int,
    random_seed: int,
) -> dict[str, torch.Tensor]:
    examples = len(images_cpu)
    stores = {
        "top1_accuracy_probability": torch.zeros(examples, dtype=torch.float64),
        "reference_top1_match_probability": torch.zeros(examples, dtype=torch.float64),
        "exact_set_probability": torch.zeros(examples, dtype=torch.float64),
        "label_coverage_probability": torch.zeros(examples, dtype=torch.float64),
        "mean_logits_top1_accuracy": torch.zeros(examples, dtype=torch.float64),
        "mean_logits_exact_set": torch.zeros(examples, dtype=torch.float64),
        "mean_boundary_margin": torch.zeros(examples, dtype=torch.float64),
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    for low in range(0, examples, batch_size):
        high = min(low + batch_size, examples)
        images = images_cpu[low:high].to(device)
        labels = labels_cpu[low:high].to(device)
        reference = reference_top2_cpu[low:high].to(device)
        reference_set = reference.sort(1).values
        base = defense.base(images)
        counts = {
            "top1": torch.zeros(len(images), device=device, dtype=torch.float64),
            "reference_top1": torch.zeros(len(images), device=device, dtype=torch.float64),
            "exact": torch.zeros(len(images), device=device, dtype=torch.float64),
            "coverage": torch.zeros(len(images), device=device, dtype=torch.float64),
        }
        logit_sum = torch.zeros(len(images), 100, device=device, dtype=torch.float64)
        completed = 0
        while completed < draws:
            chunk = min(mc_chunk, draws - completed)
            if defense.sigma > 0:
                noise = torch.randn(
                    (chunk,) + tuple(base.shape),
                    device=device,
                    dtype=base.dtype,
                    generator=generator,
                )
                queried = (base.unsqueeze(0) + defense.sigma * noise).clamp(0.0, 1.0)
                with torch.no_grad(), autocast_off(device):
                    logits = defense.anchor(queried.flatten(0, 1)).float().view(
                        chunk, len(images), 100
                    )
            else:
                if draws != 1:
                    raise AssertionError("Deterministic methods must use exactly one draw")
                with torch.no_grad(), autocast_off(device):
                    logits = defense.anchor(base).float().unsqueeze(0)
            predictions = logits.argmax(2)
            predicted_set = logits.topk(2, dim=2, largest=True, sorted=False).indices.sort(2).values
            counts["top1"] += predictions.eq(labels[None]).sum(0).double()
            counts["reference_top1"] += predictions.eq(reference[None, :, 0]).sum(0).double()
            counts["exact"] += predicted_set.eq(reference_set[None]).all(2).sum(0).double()
            counts["coverage"] += predicted_set.eq(labels[None, :, None]).any(2).sum(0).double()
            logit_sum += logits.double().sum(0)
            completed += chunk
        mean_logits = (logit_sum / float(draws)).float()
        mean_set = mean_logits.topk(2, dim=1, largest=True, sorted=False).indices.sort(1).values
        stores["top1_accuracy_probability"][low:high] = (counts["top1"] / draws).cpu()
        stores["reference_top1_match_probability"][low:high] = (
            counts["reference_top1"] / draws
        ).cpu()
        stores["exact_set_probability"][low:high] = (counts["exact"] / draws).cpu()
        stores["label_coverage_probability"][low:high] = (counts["coverage"] / draws).cpu()
        stores["mean_logits_top1_accuracy"][low:high] = mean_logits.argmax(1).eq(labels).double().cpu()
        stores["mean_logits_exact_set"][low:high] = mean_set.eq(reference_set).all(1).double().cpu()
        stores["mean_boundary_margin"][low:high] = r1.boundary_margin(
            mean_logits, reference
        ).double().cpu()
        print(f"      reevaluation batch {high}/{examples}, draws={draws}", flush=True)
    return stores


def candidate_records(
    protocol: Mapping[str, Any], pilot_indices: Sequence[int]
) -> list[dict[str, Any]]:
    output_root = Path(protocol["output_root"])
    records: list[dict[str, Any]] = []
    common_clean_hash: str | None = None
    common_index_hash: str | None = None
    common_label_hash: str | None = None
    common_reference_hash: str | None = None
    for method, attack, seed in common.expected_cells():
        directory = common.cell_dir(output_root, method, attack, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        if not candidate_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Search cell is incomplete: {directory}")
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        identity = payload_identity(
            protocol_hash=str(protocol["protocol_hash"]),
            method=method,
            attack=attack,
            search_seed=seed,
        )
        # Candidate ordering is itself checked against the parent by the search
        # evaluator; all candidates must now agree exactly with one another.
        validate_payload(payload, identity, pilot_indices)
        summary = common.read_json(summary_path)
        candidate_file_hash = common.file_sha256(candidate_path)
        if summary.get("candidate_file_sha256") != candidate_file_hash:
            raise AssertionError(f"Candidate file hash mismatch: {candidate_path}")
        hashes = payload["tensor_sha256"]
        if common_clean_hash is None:
            common_clean_hash = hashes["clean"]
            common_index_hash = hashes["sample_indices"]
            common_label_hash = hashes["labels"]
            common_reference_hash = hashes["reference_top2"]
        elif (
            hashes["clean"] != common_clean_hash
            or hashes["sample_indices"] != common_index_hash
            or hashes["labels"] != common_label_hash
            or hashes["reference_top2"] != common_reference_hash
        ):
            raise AssertionError("Search candidates do not share clean inputs/labels/references")
        records.append(
            {
                "method": method,
                "attack": attack,
                "search_seed": seed,
                "candidate_path": candidate_path,
                "candidate_file_sha256": candidate_file_hash,
                "payload": payload,
            }
        )
    return records


def cache_or_evaluate(
    *,
    path: Path,
    expected: Mapping[str, Any],
    defense: FrozenDefense,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_top2: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    draws: int,
) -> dict[str, Any]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_evaluation_payload(payload, expected)
        print(f"    RESUME {path.name}", flush=True)
        return dict(payload)
    started = time.monotonic()
    values = evaluate_images(
        defense=defense,
        images_cpu=images,
        labels_cpu=labels,
        reference_top2_cpu=reference_top2,
        device=device,
        batch_size=args.batch_size,
        mc_chunk=args.mc_chunk,
        draws=draws,
        random_seed=common.REEVAL_SEED,
    )
    payload: dict[str, Any] = {
        **expected,
        "examples": len(images),
        "draws": int(draws),
        "reeval_seed": common.REEVAL_SEED,
        "values": values,
        "wall_seconds": float(time.monotonic() - started),
    }
    add_evaluation_hash(payload)
    validate_evaluation_payload(payload, expected)
    atomic_torch_save(path, payload)
    return payload


def summary_row(record: Mapping[str, Any], evaluation: Mapping[str, Any], clean: Mapping[str, Any]) -> dict[str, Any]:
    values = evaluation["values"]
    clean_values = clean["values"]
    row: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": evaluation["protocol_hash"],
        "method": record["method"],
        "attack": record["attack"],
        "search_seed": record["search_seed"],
        "examples": int(evaluation["examples"]),
        "eval_draws": int(evaluation["draws"]),
        "reeval_seed": common.REEVAL_SEED,
        "api_budget": common.API_BUDGET,
        "epsilon_px": common.EPSILON_PX,
        "candidate_file": str(record["candidate_path"]),
        "candidate_file_sha256": record["candidate_file_sha256"],
        "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
    }
    for metric in METRICS:
        row[f"attacked_{metric}_percent" if metric != "mean_boundary_margin" else "attacked_mean_boundary_margin"] = (
            float(values[metric].mean())
            if metric == "mean_boundary_margin"
            else 100.0 * float(values[metric].mean())
        )
        row[f"clean_{metric}_percent" if metric != "mean_boundary_margin" else "clean_mean_boundary_margin"] = (
            float(clean_values[metric].mean())
            if metric == "mean_boundary_margin"
            else 100.0 * float(clean_values[metric].mean())
        )
    row["clean_to_attack_exact_set_delta_pp"] = (
        row["attacked_exact_set_probability_percent"]
        - row["clean_exact_set_probability_percent"]
    )
    return row


def family_and_selection(summaries: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_rows: list[dict[str, Any]] = []
    for method in common.METHODS:
        for attack in common.ATTACKS:
            cells = [
                row for row in summaries
                if row["method"] == method and row["attack"] == attack
            ]
            if len(cells) != len(common.SEARCH_SEEDS):
                raise AssertionError(f"Missing summary cells for {method}/{attack}")
            values = [float(row["attacked_exact_set_probability_percent"]) for row in cells]
            family_rows.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "method": method,
                    "attack": attack,
                    "seeds": "|".join(str(value) for value in common.SEARCH_SEEDS),
                    "three_seed_mean_exact_set_percent": float(np.mean(values)),
                    "three_seed_sd_pp": float(statistics.stdev(values)),
                    "seed_values_percent": "|".join(f"{value:.6f}" for value in values),
                    "selection_metric": "lowest_three_seed_mean_common_mc1000_exact_set_probability",
                }
            )
    selected: list[dict[str, Any]] = []
    for method in common.METHODS:
        candidates = [row for row in family_rows if row["method"] == method]
        candidates.sort(
            key=lambda row: (
                float(row["three_seed_mean_exact_set_percent"]),
                0 if row["attack"] == "SquareAdaptive" else 1,
            )
        )
        best, alternative = candidates
        selected.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "method": method,
                "selected_attack": best["attack"],
                "selected_three_seed_mean_exact_set_percent": best["three_seed_mean_exact_set_percent"],
                "selected_three_seed_sd_pp": best["three_seed_sd_pp"],
                "alternative_attack": alternative["attack"],
                "alternative_three_seed_mean_exact_set_percent": alternative["three_seed_mean_exact_set_percent"],
                "selected_minus_alternative_pp": float(best["three_seed_mean_exact_set_percent"])
                - float(alternative["three_seed_mean_exact_set_percent"]),
                "selection_is_pilot_only": True,
                "paper_claim_authorized": False,
            }
        )
    return family_rows, selected


def build_per_example_rows(
    records: Sequence[Mapping[str, Any]],
    evaluations: Mapping[tuple[str, str, int], Mapping[str, Any]],
    clean_cache: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        key = (record["method"], record["attack"], int(record["search_seed"]))
        evaluation = evaluations[key]
        clean = clean_cache[record["method"]]
        payload = record["payload"]
        for position in range(int(evaluation["examples"])):
            reference = [int(value) for value in payload["reference_top2"][position].tolist()]
            row: dict[str, Any] = {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": evaluation["protocol_hash"],
                "method": record["method"],
                "attack": record["attack"],
                "search_seed": record["search_seed"],
                "sample_index": int(payload["sample_indices"][position]),
                "label": int(payload["labels"][position]),
                "reference_top1": reference[0],
                "reference_set": f"{min(reference)}|{max(reference)}",
                "eval_draws": int(evaluation["draws"]),
                "reeval_seed": common.REEVAL_SEED,
                "api_budget": common.API_BUDGET,
                "epsilon_px": common.EPSILON_PX,
                "candidate_file_sha256": record["candidate_file_sha256"],
                "candidate_tensor_sha256": payload["tensor_sha256"]["adversarial"],
            }
            for metric in METRICS:
                row[f"attacked_{metric}"] = float(evaluation["values"][metric][position])
                row[f"clean_{metric}"] = float(clean["values"][metric][position])
            rows.append(row)
    return rows


def selected_attack_map(selected: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(row["method"]): str(row["selected_attack"]) for row in selected}


def pilot_comparisons(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    chosen = selected_attack_map(selected)
    attacked: dict[str, np.ndarray] = {}
    indices: list[int] | None = None
    for method in common.METHODS:
        current_indices, values = common.aligned_values(
            rows,
            method=method,
            attack=chosen[method],
            metric="attacked_exact_set_probability",
        )
        if indices is None:
            indices = current_indices
        elif current_indices != indices:
            raise AssertionError("Selected attack rows do not share sample indices")
        attacked[method] = values
    clean: dict[str, np.ndarray] = {}
    for method in common.METHODS:
        subset = [
            row for row in rows
            if row["method"] == method and row["attack"] == chosen[method]
            and int(row["search_seed"]) == common.SEARCH_SEEDS[0]
        ]
        by_index = {int(row["sample_index"]): float(row["clean_exact_set_probability"]) for row in subset}
        if sorted(by_index) != indices:
            raise AssertionError("Clean rows do not align with selected attacks")
        clean[method] = np.asarray([by_index[index] for index in indices], dtype=np.float64)
    specs = (
        ("full_minus_robust_attacked_exact_set", attacked["full"] - attacked["robust"]),
        ("full_minus_noise_only_attacked_exact_set", attacked["full"] - attacked["noise_only"]),
        ("noise_only_minus_robust_attacked_exact_set", attacked["noise_only"] - attacked["robust"]),
        ("full_sigma0_minus_robust_attacked_exact_set", attacked["full_sigma0"] - attacked["robust"]),
        ("full_minus_full_sigma0_attacked_exact_set", attacked["full"] - attacked["full_sigma0"]),
    )
    output: list[dict[str, Any]] = []
    for name, delta in specs:
        result = common.paired_bootstrap(delta)
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "comparison": name,
                "estimand": "pilot_selected_attack_three_seed_mean_per_example",
                **result,
                "exploratory": True,
                "paper_claim_authorized": False,
            }
        )
    for method in common.METHODS:
        result = common.paired_bootstrap(attacked[method] - clean[method])
        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "comparison": f"{method}_clean_to_attacked_exact_set",
                "estimand": "pilot_selected_attack_three_seed_mean_per_example_minus_clean",
                **result,
                "exploratory": True,
                "paper_claim_authorized": False,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.mc_chunk) < 1:
        raise ValueError("Batch size and MC chunk must be positive")
    if args.batch_size != common.REEVAL_BATCH_SIZE or args.mc_chunk != common.REEVAL_MC_CHUNK:
        raise ValueError("R4 requires the frozen MC=1000 batch/chunk configuration")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("GPU memory fraction must be in (0,1]")
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parent = common.load_parent_authorization(Path(protocol["parent_dir"]))
    if parent["handoff_hash"] != protocol["parent_handoff_hash"]:
        raise AssertionError("Parent handoff changed after R4 protocol freeze")
    records = candidate_records(protocol, parent["pilot_indices"])
    first = records[0]["payload"]
    images = first["clean"].detach().cpu()
    labels = first["labels"].detach().cpu()
    reference_top2 = first["reference_top2"].detach().cpu()
    standard, anchor = load_models(protocol, device)

    clean_cache: dict[str, dict[str, Any]] = {}
    for method in common.METHODS:
        spec = common.method_spec(method)
        defense = FrozenDefense(spec, standard, anchor, device)
        draws = common.REEVAL_DRAWS if spec.stochastic else 1
        expected = {
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol_hash,
            "condition": "clean",
            "method": method,
            "clean_tensor_sha256": first["tensor_sha256"]["clean"],
        }
        print(f"  Clean cache {method}, draws={draws}", flush=True)
        clean_cache[method] = cache_or_evaluate(
            path=output_dir / "cache" / f"clean_{method}.pt",
            expected=expected,
            defense=defense,
            images=images,
            labels=labels,
            reference_top2=reference_top2,
            device=device,
            args=args,
            draws=draws,
        )

    evaluations: dict[tuple[str, str, int], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        method = str(record["method"])
        attack = str(record["attack"])
        search_seed = int(record["search_seed"])
        spec = common.method_spec(method)
        defense = FrozenDefense(spec, standard, anchor, device)
        draws = common.REEVAL_DRAWS if spec.stochastic else 1
        expected = {
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol_hash,
            "condition": "attacked",
            "method": method,
            "attack": attack,
            "search_seed": search_seed,
            "candidate_file_sha256": record["candidate_file_sha256"],
            "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
        }
        print(
            f"[{position}/{len(records)}] {method}/{attack}/seed={search_seed}, draws={draws}",
            flush=True,
        )
        evaluation = cache_or_evaluate(
            path=output_dir / "cache" / f"attacked_{method}_{attack}_seed_{search_seed}.pt",
            expected=expected,
            defense=defense,
            images=record["payload"]["adversarial"].detach().cpu(),
            labels=labels,
            reference_top2=reference_top2,
            device=device,
            args=args,
            draws=draws,
        )
        evaluations[(method, attack, search_seed)] = evaluation
        summaries.append(summary_row(record, evaluation, clean_cache[method]))

    rows = build_per_example_rows(records, evaluations, clean_cache)
    family_rows, selected = family_and_selection(summaries)
    comparisons = pilot_comparisons(rows, selected)
    common.atomic_csv(output_dir / "candidate_reeval_per_example.csv", rows)
    common.atomic_csv(output_dir / "candidate_reeval_summary.csv", summaries)
    common.atomic_csv(output_dir / "attack_family_three_seed_summary.csv", family_rows)
    common.atomic_csv(output_dir / "selected_attack_envelope.csv", selected)
    common.atomic_csv(output_dir / "pilot_paired_bootstrap.csv", comparisons)

    selection_payload: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "selection_stage": "exploratory_200_example_pilot",
        "selection_rule": protocol["selection_rule"],
        "method_attack_selection": {
            row["method"]: {
                "attack": row["selected_attack"],
                "three_seed_mean_exact_set_percent": row[
                    "selected_three_seed_mean_exact_set_percent"
                ],
                "three_seed_sd_pp": row["selected_three_seed_sd_pp"],
            }
            for row in selected
        },
        "attack_configs": {
            "SquareAdaptive": protocol["adaptive"],
            "SquareRacing": protocol["racing"],
        },
        "top_k": common.TOP_K,
        "epsilon_px": common.EPSILON_PX,
        "api_budget_per_example": common.API_BUDGET,
        "square_p_init": common.SQUARE_P_INIT,
        "confirmatory_partition": {
            "examples": protocol["confirmatory_examples"],
            "indices_sha256": protocol["confirmatory_indices_sha256"],
            "outcomes_read": False,
        },
        "confirmatory_search_seeds": list(common.CONFIRMATORY_SEARCH_SEEDS),
        "defense_frozen": protocol["frozen_defense"],
        "candidate_reevaluation_draws": common.REEVAL_DRAWS,
        "candidate_reevaluation_seed": common.REEVAL_SEED,
        "no_per_example_candidate_selection": True,
        "authorized_next_stage": "disjoint_1000_example_confirmatory_attack",
        "paper_claim_authorized_by_pilot": False,
    }
    selection_payload["selection_hash"] = common.object_sha256(selection_payload)
    common.atomic_json(output_dir / "confirmatory_attack_freeze.json", selection_payload)
    final = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "search_cells": len(records),
        "per_example_rows": len(rows),
        "common_mc_draws": common.REEVAL_DRAWS,
        "family_summary": family_rows,
        "selected_attack_envelope": selected,
        "paired_bootstrap": comparisons,
        "confirmatory_attack_freeze_hash": selection_payload["selection_hash"],
        "confirmatory_partition_outcomes_read": False,
        "next_stage": "disjoint_1000_example_confirmatory_attack",
        "paper_claim_status": "pilot_is_exploratory_only",
    }
    common.atomic_json(output_dir / "final_summary.json", final)

    print("\nCIFAR-100 pilot common-MC=1000 attack-family selection", flush=True)
    for row in selected:
        print(
            f"  {row['method']:>11}: {row['selected_attack']} "
            f"Exact={float(row['selected_three_seed_mean_exact_set_percent']):.3f}% "
            f"sd={float(row['selected_three_seed_sd_pp']):.3f} pp",
            flush=True,
        )
    print("PILOT_COMPLETE=YES", flush=True)
    print("CONFIRMATORY_ATTACK_FREEZE=READY", flush=True)
    print("PAPER_CLAIM_AUTHORIZED=NO (pilot selection only)", flush=True)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
