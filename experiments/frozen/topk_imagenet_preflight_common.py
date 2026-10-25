#!/usr/bin/env python3
"""Pure, dependency-light contracts for the ImageNet Top-k R0 preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROTOCOL_ID = "imagenet_topk_preflight_r0_20260910"
NUM_CLASSES = 1000
VAL_PER_CLASS = 50
EXPECTED_TRAIN_EXAMPLES = 1_281_167
EXPECTED_VAL_EXAMPLES = 50_000
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}

# These partitions are frozen before any accuracy is computed.  Only
# preflight_sanity is read by R0.  The other partitions remain outcome-blind.
SPLIT_SPEC = OrderedDict(
    [
        ("preflight_sanity", 1),
        ("calibration_dev", 2),
        ("attack_pilot", 1),
        ("attack_confirmatory", 1),
        ("clean_confirmation", 45),
    ]
)

SPLIT_FIELDNAMES = [
    "partition",
    "target",
    "class_name",
    "within_class_slot",
    "relative_path",
]


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def class_seed(master_seed: int, class_name: str) -> int:
    payload = f"{PROTOCOL_ID}|{master_seed}|{class_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_frozen_split(
    samples_by_class: Mapping[str, Sequence[str]],
    seed: int,
) -> list[dict[str, object]]:
    """Create the outcome-blind, balanced split used by every later ImageNet stage."""
    classes = sorted(samples_by_class)
    if len(classes) != NUM_CLASSES:
        raise AssertionError(f"Expected {NUM_CLASSES} classes, got {len(classes)}")
    if sum(SPLIT_SPEC.values()) != VAL_PER_CLASS:
        raise AssertionError("SPLIT_SPEC must consume exactly 50 validation images per class")

    rows: list[dict[str, object]] = []
    for target, class_name in enumerate(classes):
        paths = sorted(str(item) for item in samples_by_class[class_name])
        if len(paths) != VAL_PER_CLASS:
            raise AssertionError(
                f"Class {class_name!r} has {len(paths)} validation images; expected {VAL_PER_CLASS}"
            )
        if len(paths) != len(set(paths)):
            raise AssertionError(f"Duplicate validation path in class {class_name!r}")
        random.Random(class_seed(seed, class_name)).shuffle(paths)
        cursor = 0
        for partition, count in SPLIT_SPEC.items():
            for slot, relative_path in enumerate(paths[cursor : cursor + count]):
                rows.append(
                    {
                        "partition": partition,
                        "target": target,
                        "class_name": class_name,
                        "within_class_slot": slot,
                        "relative_path": relative_path,
                    }
                )
            cursor += count
        if cursor != VAL_PER_CLASS:
            raise AssertionError(f"Internal split error for {class_name}")
    validate_frozen_split(rows)
    return rows


def validate_frozen_split(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) != EXPECTED_VAL_EXAMPLES:
        raise AssertionError(f"Expected {EXPECTED_VAL_EXAMPLES} split rows, got {len(rows)}")
    paths = [str(row["relative_path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise AssertionError("Frozen split contains overlapping/duplicated image paths")

    partitions = Counter(str(row["partition"]) for row in rows)
    expected_partitions = {
        name: per_class * NUM_CLASSES for name, per_class in SPLIT_SPEC.items()
    }
    if dict(partitions) != expected_partitions:
        raise AssertionError(
            f"Partition sizes differ: expected={expected_partitions}, got={dict(partitions)}"
        )

    class_names = sorted({str(row["class_name"]) for row in rows})
    targets = sorted({int(row["target"]) for row in rows})
    if len(class_names) != NUM_CLASSES or targets != list(range(NUM_CLASSES)):
        raise AssertionError("Frozen split does not contain the canonical 1000-class mapping")

    per_partition_class: dict[str, Counter[int]] = {
        name: Counter() for name in SPLIT_SPEC
    }
    mapping: dict[str, int] = {}
    for row in rows:
        partition = str(row["partition"])
        class_name = str(row["class_name"])
        target = int(row["target"])
        if partition not in SPLIT_SPEC:
            raise AssertionError(f"Unknown partition {partition!r}")
        previous = mapping.setdefault(class_name, target)
        if previous != target:
            raise AssertionError(f"Inconsistent target mapping for {class_name}")
        per_partition_class[partition][target] += 1

    for partition, per_class in SPLIT_SPEC.items():
        counter = per_partition_class[partition]
        if set(counter) != set(range(NUM_CLASSES)) or set(counter.values()) != {per_class}:
            raise AssertionError(f"Partition {partition} is not class-balanced")

    canonical_rows = [
        {
            "partition": str(row["partition"]),
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "within_class_slot": int(row["within_class_slot"]),
            "relative_path": str(row["relative_path"]),
        }
        for row in rows
    ]
    return {
        "rows": len(rows),
        "partitions": expected_partitions,
        "classes": NUM_CLASSES,
        "split_sha256": stable_hash(canonical_rows),
        "class_mapping_sha256": stable_hash(
            [{"class_name": name, "target": mapping[name]} for name in sorted(mapping)]
        ),
    }


def partition_rows(
    rows: Iterable[Mapping[str, object]], partition: str
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if str(row["partition"]) == partition:
            selected.append(
                {
                    "partition": partition,
                    "target": int(row["target"]),
                    "class_name": str(row["class_name"]),
                    "within_class_slot": int(row["within_class_slot"]),
                    "relative_path": str(row["relative_path"]),
                }
            )
    return selected


def same_rows(left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]]) -> bool:
    def canonical(rows: Sequence[Mapping[str, object]]) -> list[tuple[object, ...]]:
        return [
            (
                str(row["partition"]),
                int(row["target"]),
                str(row["class_name"]),
                int(row["within_class_slot"]),
                str(row["relative_path"]),
            )
            for row in rows
        ]

    return canonical(left) == canonical(right)


def human_bytes(value: int) -> str:
    number = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or suffix == "TiB":
            return f"{number:.2f} {suffix}"
        number /= 1024.0
    raise AssertionError("unreachable")

