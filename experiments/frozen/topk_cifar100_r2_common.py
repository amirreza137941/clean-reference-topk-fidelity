#!/usr/bin/env python3
"""Shared, dependency-light protocol utilities for CIFAR-100 R2.

The scientific invariant enforced here is that *one immutable split* is used
to train the standard reference, select the adversarial anchor, and select the
Dual defense.  The official CIFAR-100 test set is never exposed until a stage
has selected its checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "cifar100_resnet20_external_validity_r2_20260903"
DEFAULT_DATA = Path("./datasets/cifar-100-python")
DEFAULT_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_SEED = 9103
DEFAULT_VAL_SIZE = 5000
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def resolve_torchvision_root(path: Path) -> Path:
    """Return the directory that torchvision expects as ``root``."""
    path = path.expanduser().resolve()
    if path.name == "cifar-100-python" and (path / "train").is_file():
        return path.parent
    nested = path / "cifar-100-python"
    if (nested / "train").is_file() and (nested / "test").is_file():
        return path
    raise FileNotFoundError(
        f"CIFAR-100 batches not found under '{path}' or '{nested}'."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[int]) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def labels_sha256(values: Sequence[int]) -> str:
    array = np.asarray(values, dtype="<i2")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def stratified_train_val_indices(
    targets: Sequence[int], val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    if len(targets) != 50_000:
        raise ValueError(f"Expected 50000 CIFAR-100 training labels, got {len(targets)}")
    if val_size <= 0 or val_size >= len(targets) or val_size % 100 != 0:
        raise ValueError("Validation size must be divisible by 100 and below 50000")
    targets_array = np.asarray(targets)
    if set(Counter(int(x) for x in targets_array).values()) != {500}:
        raise ValueError("Expected exactly 500 training examples per CIFAR-100 class")
    rng = np.random.default_rng(seed)
    per_class = val_size // 100
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in range(100):
        indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(indices)
        validation_indices.extend(int(x) for x in indices[:per_class])
        train_indices.extend(int(x) for x in indices[per_class:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def _validate_split_payload(
    payload: Mapping[str, Any], targets: Sequence[int], seed: int, val_size: int
) -> tuple[list[int], list[int]]:
    required = {
        "protocol_id",
        "dataset",
        "seed",
        "validation_size",
        "train_indices",
        "validation_indices",
        "train_indices_sha256",
        "validation_indices_sha256",
        "labels_sha256",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Frozen split misses fields: {sorted(missing)}")
    if payload["protocol_id"] != PROTOCOL_ID or payload["dataset"] != "CIFAR-100":
        raise ValueError("Frozen split belongs to another protocol or dataset")
    if int(payload["seed"]) != int(seed) or int(payload["validation_size"]) != val_size:
        raise ValueError("Frozen split seed/size differs from the requested protocol")
    train_indices = [int(x) for x in payload["train_indices"]]
    validation_indices = [int(x) for x in payload["validation_indices"]]
    if len(train_indices) != len(targets) - val_size or len(validation_indices) != val_size:
        raise ValueError("Frozen split has incorrect train/validation lengths")
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("Frozen training indices contain duplicates")
    if len(set(validation_indices)) != len(validation_indices):
        raise ValueError("Frozen validation indices contain duplicates")
    if set(train_indices).intersection(validation_indices):
        raise ValueError("Frozen training and validation indices overlap")
    if set(train_indices).union(validation_indices) != set(range(len(targets))):
        raise ValueError("Frozen split does not partition all 50000 training examples")
    if sequence_sha256(train_indices) != payload["train_indices_sha256"]:
        raise ValueError("Frozen training-index hash mismatch")
    if sequence_sha256(validation_indices) != payload["validation_indices_sha256"]:
        raise ValueError("Frozen validation-index hash mismatch")
    if labels_sha256(targets) != payload["labels_sha256"]:
        raise ValueError("Dataset-label hash differs from the split manifest")
    val_counts = Counter(int(targets[i]) for i in validation_indices)
    train_counts = Counter(int(targets[i]) for i in train_indices)
    if set(val_counts) != set(range(100)) or set(val_counts.values()) != {val_size // 100}:
        raise ValueError("Frozen validation split is not class balanced")
    expected_train_per_class = 500 - val_size // 100
    if set(train_counts) != set(range(100)) or set(train_counts.values()) != {expected_train_per_class}:
        raise ValueError("Frozen training split is not class balanced")
    return train_indices, validation_indices


def create_or_validate_frozen_split(
    path: Path, targets: Sequence[int], seed: int = DEFAULT_SEED,
    val_size: int = DEFAULT_VAL_SIZE,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_file():
        payload = read_json(path)
        _validate_split_payload(payload, targets, seed, val_size)
        return payload
    train_indices, validation_indices = stratified_train_val_indices(
        targets, val_size, seed
    )
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "dataset": "CIFAR-100",
        "seed": int(seed),
        "total_training_examples": len(targets),
        "training_size": len(train_indices),
        "validation_size": len(validation_indices),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_indices_sha256": sequence_sha256(train_indices),
        "validation_indices_sha256": sequence_sha256(validation_indices),
        "labels_sha256": labels_sha256(targets),
        "rule": "fixed class-stratified 450/50 train/validation examples per class",
    }
    atomic_json(path, payload)
    _validate_split_payload(payload, targets, seed, val_size)
    return payload


def load_frozen_split(
    path: Path, targets: Sequence[int], seed: int = DEFAULT_SEED,
    val_size: int = DEFAULT_VAL_SIZE,
) -> tuple[list[int], list[int], dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen split not found: {path}. Run the R2 preflight first."
        )
    payload = read_json(path)
    train_indices, validation_indices = _validate_split_payload(
        payload, targets, seed, val_size
    )
    return train_indices, validation_indices, payload


def balanced_subset_from_indices(
    all_targets: Sequence[int], candidate_indices: Sequence[int], size: int
) -> list[int]:
    if size <= 0 or size % 100 != 0:
        raise ValueError("Balanced subset size must be positive and divisible by 100")
    per_class = size // 100
    selected: list[int] = []
    counts = [0] * 100
    for raw_index in candidate_indices:
        index = int(raw_index)
        label = int(all_targets[index])
        if counts[label] < per_class:
            selected.append(index)
            counts[label] += 1
    if any(count != per_class for count in counts):
        raise ValueError(f"Could not form balanced subset; counts={counts}")
    return selected
