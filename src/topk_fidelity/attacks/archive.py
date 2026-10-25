"""Original per-image candidate archives used during attack search."""

from __future__ import annotations

import torch


def _archive_accepted_candidates(
    archive: torch.Tensor,
    archive_scores: torch.Tensor,
    candidate: torch.Tensor,
    candidate_score: torch.Tensor,
    accepted: torch.Tensor,
) -> None:
    worst_slots = archive_scores.argmax(0)
    samples = torch.arange(len(candidate), device=candidate.device)
    worst_scores = archive_scores[worst_slots, samples]
    replace = accepted & (candidate_score < worst_scores)
    replace_samples = samples[replace]
    replace_slots = worst_slots[replace]
    archive[replace_slots, replace_samples] = candidate[replace]
    archive_scores[replace_slots, replace_samples] = candidate_score[replace]


def _insert_racing_candidate(
    archive: torch.Tensor,
    archive_scores: torch.Tensor,
    candidate: torch.Tensor,
    candidate_score: torch.Tensor,
    filled: int,
) -> int:
    pool_size = int(archive.shape[0])
    if filled < pool_size:
        archive[filled] = candidate
        archive_scores[filled] = candidate_score
        return filled + 1
    worst_slots = archive_scores.argmax(0)
    samples = torch.arange(len(candidate), device=candidate.device)
    worst_scores = archive_scores[worst_slots, samples]
    replace = candidate_score < worst_scores
    replace_samples = samples[replace]
    replace_slots = worst_slots[replace]
    archive[replace_slots, replace_samples] = candidate[replace]
    archive_scores[replace_slots, replace_samples] = candidate_score[replace]
    return filled
