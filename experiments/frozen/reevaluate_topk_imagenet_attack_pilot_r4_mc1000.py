#!/usr/bin/env python3
"""Common-MC candidate reevaluation and attack-family freeze for ImageNet R4."""

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

import topk_imagenet_attack_pilot_r4_common as common
from prepare_topk_imagenet_attack_pilot_r4 import validate_input_payload
from search_topk_imagenet_attack_pilot_r4 import (
    FrozenDefense,
    atomic_torch_save,
    identity,
    validate_candidate_payload,
)
import calibrate_topk_imagenet_defense_r2 as r2


METRICS = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "label_coverage_k5_probability",
    "exact_set_k5_probability",
    "exact_set_k2_probability",
    "mean_logits_top1_accuracy",
    "mean_logits_exact_set_k5",
    "mean_logits_exact_set_k2",
    "mean_boundary_margin_k5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=common.REEVAL_BATCH_SIZE)
    parser.add_argument("--mc-chunk", type=int, default=common.REEVAL_MC_CHUNK)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def autocast_off(device: torch.device):
    return torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else contextlib.nullcontext()


def boundary_margin(logits: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    selected_min = logits.gather(1, reference).min(1).values
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference, False)
    return selected_min - logits.masked_fill(~outside, -torch.inf).max(1).values


def evaluation_hash_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in payload.items() if key not in {"values", "evaluation_hash"}}
    result["value_sha256"] = {
        key: common.tensor_sha256(payload["values"][key]) for key in sorted(payload["values"])
    }
    return result


def add_evaluation_hash(payload: dict[str, Any]) -> None:
    payload["value_sha256"] = {
        key: common.tensor_sha256(payload["values"][key]) for key in sorted(payload["values"])
    }
    payload["evaluation_hash"] = common.object_sha256(evaluation_hash_input(payload))


def validate_evaluation(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Evaluation identity mismatch: {key}")
    values = payload.get("values", {})
    if set(values) != set(METRICS):
        raise AssertionError("Evaluation metric set changed")
    examples = int(payload.get("examples", -1))
    for key in METRICS:
        tensor = values[key].detach().cpu()
        if tensor.shape != (examples,) or not torch.isfinite(tensor).all():
            raise AssertionError(f"Invalid evaluation tensor: {key}")
        if key != "mean_boundary_margin_k5" and (float(tensor.min()) < -1e-12 or float(tensor.max()) > 1 + 1e-12):
            raise AssertionError(f"Evaluation probability outside [0,1]: {key}")
        if common.tensor_sha256(tensor) != payload.get("value_sha256", {}).get(key):
            raise AssertionError(f"Evaluation tensor hash mismatch: {key}")
    if common.object_sha256(evaluation_hash_input(payload)) != payload.get("evaluation_hash"):
        raise AssertionError("Evaluation payload hash mismatch")


def evaluate_images(
    *, defense: FrozenDefense, images_cpu: torch.Tensor, labels_cpu: torch.Tensor,
    reference_cpu: torch.Tensor, device: torch.device, batch_size: int,
    mc_chunk: int, draws: int, random_seed: int,
) -> dict[str, torch.Tensor]:
    examples = len(images_cpu)
    stores = {key: torch.zeros(examples, dtype=torch.float64) for key in METRICS}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    for low in range(0, examples, batch_size):
        high = min(low + batch_size, examples)
        images = images_cpu[low:high].to(device)
        labels = labels_cpu[low:high].to(device)
        reference = reference_cpu[low:high].to(device)
        ref5_set = reference.sort(1).values
        ref2_set = reference[:, : common.SECONDARY_K].sort(1).values
        base = defense.base(images)
        counts = {key: torch.zeros(len(images), dtype=torch.float64, device=device) for key in (
            "top1", "reference_top1", "coverage5", "exact5", "exact2"
        )}
        logit_sum = torch.zeros(len(images), 1000, dtype=torch.float64, device=device)
        completed = 0
        while completed < draws:
            chunk = min(mc_chunk, draws - completed)
            if defense.sigma > 0:
                noise = torch.randn((chunk,) + tuple(base.shape), device=device, dtype=base.dtype, generator=generator)
                queried = (base.unsqueeze(0) + defense.sigma * noise).clamp(0.0, 1.0)
                with torch.no_grad(), autocast_off(device):
                    logits = defense.robust(queried.flatten(0, 1)).float().view(chunk, len(images), 1000)
            else:
                if draws != 1:
                    raise AssertionError("Deterministic methods must use one reevaluation draw")
                with torch.no_grad(), autocast_off(device):
                    logits = defense.robust(base).float().unsqueeze(0)
            predictions = logits.argmax(2)
            set5 = logits.topk(common.PRIMARY_K, dim=2, sorted=False).indices.sort(2).values
            set2 = logits.topk(common.SECONDARY_K, dim=2, sorted=False).indices.sort(2).values
            counts["top1"] += predictions.eq(labels[None]).sum(0).double()
            counts["reference_top1"] += predictions.eq(reference[None, :, 0]).sum(0).double()
            counts["coverage5"] += set5.eq(labels[None, :, None]).any(2).sum(0).double()
            counts["exact5"] += set5.eq(ref5_set[None]).all(2).sum(0).double()
            counts["exact2"] += set2.eq(ref2_set[None]).all(2).sum(0).double()
            logit_sum += logits.double().sum(0)
            completed += chunk
        mean_logits = (logit_sum / float(draws)).float()
        mean5 = mean_logits.topk(common.PRIMARY_K, dim=1, sorted=False).indices.sort(1).values
        mean2 = mean_logits.topk(common.SECONDARY_K, dim=1, sorted=False).indices.sort(1).values
        stores["top1_accuracy_probability"][low:high] = (counts["top1"] / draws).cpu()
        stores["reference_top1_match_probability"][low:high] = (counts["reference_top1"] / draws).cpu()
        stores["label_coverage_k5_probability"][low:high] = (counts["coverage5"] / draws).cpu()
        stores["exact_set_k5_probability"][low:high] = (counts["exact5"] / draws).cpu()
        stores["exact_set_k2_probability"][low:high] = (counts["exact2"] / draws).cpu()
        stores["mean_logits_top1_accuracy"][low:high] = mean_logits.argmax(1).eq(labels).double().cpu()
        stores["mean_logits_exact_set_k5"][low:high] = mean5.eq(ref5_set).all(1).double().cpu()
        stores["mean_logits_exact_set_k2"][low:high] = mean2.eq(ref2_set).all(1).double().cpu()
        stores["mean_boundary_margin_k5"][low:high] = boundary_margin(mean_logits, reference).double().cpu()
        print(f"      reevaluation {high}/{examples}, draws={draws}", flush=True)
    return stores


def cache_or_evaluate(
    *, path: Path, expected: Mapping[str, Any], defense: FrozenDefense,
    images: torch.Tensor, labels: torch.Tensor, reference: torch.Tensor,
    device: torch.device, args: argparse.Namespace, draws: int,
) -> dict[str, Any]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_evaluation(payload, expected)
        print(f"    RESUME {path.name}", flush=True)
        return dict(payload)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        **expected,
        "examples": len(images),
        "draws": int(draws),
        "reeval_seed": common.REEVAL_SEED,
        "values": evaluate_images(
            defense=defense, images_cpu=images, labels_cpu=labels,
            reference_cpu=reference, device=device, batch_size=args.batch_size,
            mc_chunk=args.mc_chunk, draws=draws, random_seed=common.REEVAL_SEED,
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    add_evaluation_hash(payload)
    validate_evaluation(payload, expected)
    atomic_torch_save(path, payload)
    return payload


def load_records(protocol: Mapping[str, Any], pilot: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(protocol["paths"]["output_root"])
    positions = [int(value) for value in pilot["pilot_positions"]]
    paths = list(pilot["relative_paths"])
    records: list[dict[str, Any]] = []
    clean_hash = common.tensor_sha256(pilot["clean"])
    label_hash = common.tensor_sha256(pilot["labels"])
    reference_hash = common.tensor_sha256(pilot["reference_top5"])
    position_hash = common.tensor_sha256(torch.tensor(positions, dtype=torch.long))
    for method, attack, seed in common.expected_cells():
        directory = common.cell_dir(root, method, attack, seed)
        candidate_path, summary_path = directory / "candidate.pt", directory / "search_summary.json"
        if not candidate_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete search cell: {directory}")
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        expected = identity(protocol, method, attack, seed, str(pilot["payload_hash"]))
        validate_candidate_payload(payload, expected, positions, paths)
        summary = common.read_json(summary_path)
        file_hash = common.file_sha256(candidate_path)
        if summary.get("candidate_file_sha256") != file_hash or summary.get("candidate_payload_hash") != payload["payload_hash"]:
            raise AssertionError(f"Search summary mismatch: {directory}")
        if payload["tensor_sha256"]["clean"] != clean_hash:
            raise AssertionError("Candidate clean input differs from frozen pilot")
        if payload["tensor_sha256"]["labels"] != label_hash:
            raise AssertionError("Candidate labels differ from frozen pilot")
        if payload["tensor_sha256"]["reference_top5"] != reference_hash:
            raise AssertionError("Candidate references differ from frozen pilot")
        if payload["tensor_sha256"]["pilot_positions"] != position_hash:
            raise AssertionError("Candidate pilot positions differ from frozen pilot")
        records.append({
            "method": method, "attack": attack, "search_seed": seed,
            "candidate_path": candidate_path, "candidate_file_sha256": file_hash,
            "payload": payload,
        })
    return records


def summary_row(record: Mapping[str, Any], attacked: Mapping[str, Any], clean: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": attacked["protocol_hash"],
        "method": record["method"], "attack": record["attack"],
        "search_seed": record["search_seed"], "examples": attacked["examples"],
        "eval_draws": attacked["draws"], "reeval_seed": common.REEVAL_SEED,
        "api_budget": common.API_BUDGET, "epsilon_px": common.EPSILON_PX,
        "candidate_file_sha256": record["candidate_file_sha256"],
        "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
    }
    for metric in METRICS:
        suffix = "" if metric.startswith("mean_") else "_percent"
        scale = 1.0 if metric == "mean_boundary_margin_k5" else 100.0
        row[f"attacked_{metric}{suffix}"] = scale * float(attacked["values"][metric].mean())
        row[f"clean_{metric}{suffix}"] = scale * float(clean["values"][metric].mean())
    row["clean_to_attack_exact_set_k5_delta_pp"] = (
        row["attacked_exact_set_k5_probability_percent"] - row["clean_exact_set_k5_probability_percent"]
    )
    return row


def build_rows(
    records: Sequence[Mapping[str, Any]], evaluations: Mapping[tuple[str, str, int], Mapping[str, Any]],
    clean_cache: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        key = (record["method"], record["attack"], record["search_seed"])
        attacked, clean, source = evaluations[key], clean_cache[record["method"]], record["payload"]
        for position in range(common.PILOT_EXAMPLES):
            reference = [int(value) for value in source["reference_top5"][position]]
            row: dict[str, Any] = {
                "protocol_id": common.PROTOCOL_ID, "protocol_hash": attacked["protocol_hash"],
                "method": record["method"], "attack": record["attack"],
                "search_seed": record["search_seed"], "pilot_position": position,
                "relative_path": source["relative_paths"][position],
                "label": int(source["labels"][position]), "reference_top1": reference[0],
                "reference_set_k5": "|".join(str(value) for value in sorted(reference)),
                "eval_draws": attacked["draws"], "api_budget": common.API_BUDGET,
                "candidate_file_sha256": record["candidate_file_sha256"],
            }
            for metric in METRICS:
                row[f"attacked_{metric}"] = float(attacked["values"][metric][position])
                row[f"clean_{metric}"] = float(clean["values"][metric][position])
            rows.append(row)
    return rows


def aligned_seed_mean(rows: Sequence[Mapping[str, Any]], method: str, attack: str, field: str) -> np.ndarray:
    matrix = []
    for seed in common.SEARCH_SEEDS:
        selected = [row for row in rows if row["method"] == method and row["attack"] == attack and int(row["search_seed"]) == seed]
        selected.sort(key=lambda row: int(row["pilot_position"]))
        if len(selected) != common.PILOT_EXAMPLES or [int(row["pilot_position"]) for row in selected] != list(range(common.PILOT_EXAMPLES)):
            raise AssertionError(f"Rows are incomplete for {method}/{attack}/{seed}")
        matrix.append([float(row[field]) for row in selected])
    return np.asarray(matrix, dtype=np.float64).mean(0)


def family_selection(summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]):
    families: list[dict[str, Any]] = []
    for method in common.METHODS:
        for attack in common.ATTACKS:
            cells = [row for row in summaries if row["method"] == method and row["attack"] == attack]
            values = [float(row["attacked_exact_set_k5_probability_percent"]) for row in cells]
            if len(values) != len(common.SEARCH_SEEDS):
                raise AssertionError(f"Missing family cells for {method}/{attack}")
            families.append({
                "protocol_id": common.PROTOCOL_ID, "method": method, "attack": attack,
                "seeds": "|".join(map(str, common.SEARCH_SEEDS)),
                "three_seed_mean_exact_set_k5_percent": float(np.mean(values)),
                "three_seed_sd_pp": float(statistics.stdev(values)),
                "seed_values_percent": "|".join(f"{value:.6f}" for value in values),
            })
    selected: list[dict[str, Any]] = []
    family_ci: list[dict[str, Any]] = []
    for method in common.METHODS:
        adaptive = aligned_seed_mean(rows, method, "SquareAdaptive", "attacked_exact_set_k5_probability")
        racing = aligned_seed_mean(rows, method, "SquareRacing", "attacked_exact_set_k5_probability")
        ci = common.paired_bootstrap(adaptive - racing)
        family_ci.append({
            "protocol_id": common.PROTOCOL_ID, "method": method,
            "comparison": "SquareAdaptive_minus_SquareRacing_exact_set_k5",
            **ci, "exploratory": True, "paper_claim_authorized": False,
        })
        candidates = sorted(
            [row for row in families if row["method"] == method],
            key=lambda row: (float(row["three_seed_mean_exact_set_k5_percent"]), 0 if row["attack"] == "SquareAdaptive" else 1),
        )
        best, alternative = candidates
        selected.append({
            "protocol_id": common.PROTOCOL_ID, "method": method,
            "selected_attack": best["attack"],
            "selected_three_seed_mean_exact_set_k5_percent": best["three_seed_mean_exact_set_k5_percent"],
            "selected_three_seed_sd_pp": best["three_seed_sd_pp"],
            "alternative_attack": alternative["attack"],
            "alternative_three_seed_mean_exact_set_k5_percent": alternative["three_seed_mean_exact_set_k5_percent"],
            "selected_minus_alternative_pp": float(best["three_seed_mean_exact_set_k5_percent"]) - float(alternative["three_seed_mean_exact_set_k5_percent"]),
            "pilot_only": True, "paper_claim_authorized": False,
        })
    return families, selected, family_ci


def exploratory_comparisons(rows: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    attack_map = {str(row["method"]): str(row["selected_attack"]) for row in selected}
    attacked = {
        method: aligned_seed_mean(rows, method, attack_map[method], "attacked_exact_set_k5_probability")
        for method in common.METHODS
    }
    specs = (
        ("full_minus_robust_attacked_exact_set_k5", attacked["full"] - attacked["robust"]),
        ("full_minus_noise_only_attacked_exact_set_k5", attacked["full"] - attacked["noise_only"]),
        ("noise_only_minus_robust_attacked_exact_set_k5", attacked["noise_only"] - attacked["robust"]),
        ("full_sigma0_minus_robust_attacked_exact_set_k5", attacked["full_sigma0"] - attacked["robust"]),
        ("full_minus_full_sigma0_attacked_exact_set_k5", attacked["full"] - attacked["full_sigma0"]),
    )
    return [{
        "protocol_id": common.PROTOCOL_ID, "comparison": name,
        "estimand": "pilot-selected-attack three-seed mean per image", **common.paired_bootstrap(delta),
        "exploratory": True, "paper_claim_authorized": False,
    } for name, delta in specs]


def main() -> None:
    args = parse_args()
    if args.batch_size != common.REEVAL_BATCH_SIZE or args.mc_chunk != common.REEVAL_MC_CHUNK:
        raise ValueError("R4 requires the frozen reevaluation batch/chunk sizes")
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    output_dir = args.output_dir.expanduser().resolve()
    expected_dir = (Path(protocol["paths"]["output_root"]) / "common_mc1000").resolve()
    if output_dir != expected_dir:
        raise AssertionError("Reevaluation output directory differs from protocol")
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = Path(protocol["paths"]["output_root"]) / "pilot_inputs.pt"
    pilot = torch.load(pilot_path, map_location="cpu", weights_only=False)
    validate_input_payload(pilot, protocol)
    records = load_records(protocol, pilot)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    standard, robust, _audit = r2.load_models(
        Path(protocol["checkpoints"]["standard"]["path"]),
        Path(protocol["checkpoints"]["robust"]["path"]), device,
    )
    images, labels, reference = pilot["clean"], pilot["labels"], pilot["reference_top5"]
    cache_dir = output_dir / "cache"
    clean_cache: dict[str, Mapping[str, Any]] = {}
    for method in common.METHODS:
        spec = common.method_spec(method)
        defense = FrozenDefense(spec, standard, robust, device)
        draws = common.REEVAL_DRAWS if spec.stochastic else 1
        expected = {
            "protocol_id": common.PROTOCOL_ID, "protocol_hash": protocol_hash,
            "condition": "clean", "method": method,
            "clean_tensor_sha256": common.tensor_sha256(images),
            "label_tensor_sha256": common.tensor_sha256(labels),
            "reference_tensor_sha256": common.tensor_sha256(reference),
            "pilot_payload_hash": pilot["payload_hash"],
        }
        print(f"  Clean {method}, draws={draws}", flush=True)
        clean_cache[method] = cache_or_evaluate(
            path=cache_dir / f"clean_{method}.pt", expected=expected, defense=defense,
            images=images, labels=labels, reference=reference, device=device, args=args, draws=draws,
        )
    evaluations: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for number, record in enumerate(records, 1):
        method, attack, seed = record["method"], record["attack"], record["search_seed"]
        spec = common.method_spec(method)
        defense = FrozenDefense(spec, standard, robust, device)
        draws = common.REEVAL_DRAWS if spec.stochastic else 1
        expected = {
            "protocol_id": common.PROTOCOL_ID, "protocol_hash": protocol_hash,
            "condition": "attacked", "method": method, "attack": attack,
            "search_seed": seed, "candidate_file_sha256": record["candidate_file_sha256"],
            "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
        }
        print(f"[{number}/{len(records)}] {method}/{attack}/seed={seed}, draws={draws}", flush=True)
        evaluated = cache_or_evaluate(
            path=cache_dir / f"attacked_{method}_{attack}_seed_{seed}.pt",
            expected=expected, defense=defense, images=record["payload"]["adversarial"],
            labels=labels, reference=reference, device=device, args=args, draws=draws,
        )
        evaluations[(method, attack, seed)] = evaluated
        summaries.append(summary_row(record, evaluated, clean_cache[method]))
    rows = build_rows(records, evaluations, clean_cache)
    families, selected, family_ci = family_selection(summaries, rows)
    comparisons = exploratory_comparisons(rows, selected)
    common.atomic_csv(output_dir / "candidate_reeval_per_example.csv", rows)
    common.atomic_csv(output_dir / "candidate_reeval_summary.csv", summaries)
    common.atomic_csv(output_dir / "attack_family_three_seed_summary.csv", families)
    common.atomic_csv(output_dir / "attack_family_paired_bootstrap.csv", family_ci)
    common.atomic_csv(output_dir / "selected_attack_envelope.csv", selected)
    common.atomic_csv(output_dir / "pilot_exploratory_comparisons.csv", comparisons)
    freeze: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID, "protocol_hash": protocol_hash, "status": "PASS",
        "selection_stage": "exploratory_outcome_blind_100_image_pilot",
        "selection_rule": protocol["selection_rule"],
        "method_attack_selection": {
            row["method"]: {
                "attack": row["selected_attack"],
                "three_seed_mean_exact_set_k5_percent": row["selected_three_seed_mean_exact_set_k5_percent"],
                "three_seed_sd_pp": row["selected_three_seed_sd_pp"],
            } for row in selected
        },
        "attack_configs": {"SquareAdaptive": protocol["adaptive"], "SquareRacing": protocol["racing"]},
        "primary_k": common.PRIMARY_K, "secondary_k": common.SECONDARY_K,
        "epsilon_px": common.EPSILON_PX, "api_budget_per_example": common.API_BUDGET,
        "square_p_init": common.SQUARE_P_INIT,
        "confirmatory_partition": {
            "examples": common.CONFIRMATORY_EXAMPLES,
            "rows_sha256": protocol["confirmatory_rows_sha256"], "outcomes_read": False,
        },
        "confirmatory_search_seeds": list(common.CONFIRMATORY_SEARCH_SEEDS),
        "frozen_defense": common.FROZEN_DEFENSE,
        "candidate_reevaluation_draws": common.REEVAL_DRAWS,
        "candidate_reevaluation_seed": common.REEVAL_SEED,
        "no_per_example_candidate_selection": True,
        "authorized_next_stage": "ImageNet_R5_disjoint_1000_image_confirmatory_attack",
        "paper_claim_authorized_by_pilot": False,
    }
    freeze["selection_hash"] = common.object_sha256(freeze)
    common.write_or_validate_json(output_dir / "confirmatory_attack_freeze.json", freeze)
    final = {
        "protocol_id": common.PROTOCOL_ID, "protocol_hash": protocol_hash, "status": "PASS",
        "search_cells": len(records), "per_example_rows": len(rows),
        "common_mc_draws": common.REEVAL_DRAWS, "family_summary": families,
        "selected_attack_envelope": selected, "attack_family_bootstrap": family_ci,
        "exploratory_comparisons": comparisons,
        "confirmatory_attack_freeze_hash": freeze["selection_hash"],
        "confirmatory_partition_outcomes_read": False,
        "next_stage": freeze["authorized_next_stage"],
        "paper_claim_status": "PILOT_ONLY_NO_ADVERSARIAL_CLAIM",
    }
    common.write_or_validate_json(output_dir / "final_summary.json", final)
    print("\nImageNet R4 pilot common-MC=1000 attack-family selection", flush=True)
    for row in selected:
        print(f"  {row['method']:>11}: {row['selected_attack']} "
              f"Exact5={row['selected_three_seed_mean_exact_set_k5_percent']:.3f}% "
              f"sd={row['selected_three_seed_sd_pp']:.3f} pp", flush=True)
    print("PILOT_COMPLETE=YES", flush=True)
    print("CONFIRMATORY_ATTACK_FREEZE=READY", flush=True)
    print("PAPER_CLAIM_AUTHORIZED=NO (pilot selection only)", flush=True)


if __name__ == "__main__":
    main()
