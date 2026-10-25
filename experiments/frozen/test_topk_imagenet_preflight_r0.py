#!/usr/bin/env python3
"""Server-side self-tests for the offline ImageNet R0 preflight package."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from PIL import Image

import preflight_topk_imagenet_r0 as preflight
from topk_imagenet_preflight_common import (
    NUM_CLASSES,
    PROTOCOL_ID,
    SPLIT_SPEC,
    atomic_csv,
    atomic_json,
    build_frozen_split,
    read_csv,
    read_json,
    same_rows,
    stable_hash,
    validate_frozen_split,
)


DEFAULT_DATA_ROOT = Path("./datasets/imagenet")
DEFAULT_RESNET50 = Path(
    "./models/resnet50/imagenet/resnet50_imagenet.pth"
)
DEFAULT_VIT_B16 = Path(
    "./models/vit/imagenet/vit_b16_imagenet.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test ImageNet Top-k R0 preflight")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--resnet50-checkpoint", type=Path, default=DEFAULT_RESNET50)
    parser.add_argument("--vit-b16-checkpoint", type=Path, default=DEFAULT_VIT_B16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def test_offline_source_policy() -> None:
    project = Path(__file__).resolve().parent
    names = [
        "topk_imagenet_preflight_common.py",
        "preflight_topk_imagenet_r0.py",
        "audit_topk_imagenet_preflight_r0.py",
        "run_topk_imagenet_preflight_r0.py",
    ]
    forbidden = (
        "download=True",
        "pretrained=True",
        "torch.hub",
        "requests.",
        "urllib.request",
        "curl",
        "wget",
    )
    for name in names:
        path = project / name
        if not path.is_file():
            raise FileNotFoundError(path)
        compact = path.read_text(encoding="utf-8").replace(" ", "")
        for marker in forbidden:
            if marker.replace(" ", "") in compact:
                raise AssertionError(f"Forbidden network/download marker {marker!r} in {name}")


def synthetic_samples() -> dict[str, list[str]]:
    return {
        f"n{class_id:08d}": [
            f"val/n{class_id:08d}/image_{index:02d}.JPEG" for index in range(50)
        ]
        for class_id in range(NUM_CLASSES)
    }


def test_split_contract() -> None:
    samples = synthetic_samples()
    left = build_frozen_split(samples, 9102026)
    right = build_frozen_split(samples, 9102026)
    different = build_frozen_split(samples, 9102027)
    summary = validate_frozen_split(left)
    if not same_rows(left, right):
        raise AssertionError("Frozen split is not deterministic")
    if same_rows(left, different):
        raise AssertionError("Changing the split seed did not change assignments")
    if summary["partitions"] != {
        name: count * NUM_CLASSES for name, count in SPLIT_SPEC.items()
    }:
        raise AssertionError("Partition cardinalities are incorrect")


def test_atomic_roundtrip() -> None:
    rows = build_frozen_split(synthetic_samples(), 9102026)[:20]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        json_path = root / "sample.json"
        csv_path = root / "sample.csv"
        payload = {"protocol_id": PROTOCOL_ID, "digest": stable_hash(rows)}
        atomic_json(json_path, payload)
        atomic_csv(
            csv_path,
            rows,
            ["partition", "target", "class_name", "within_class_slot", "relative_path"],
        )
        if read_json(json_path) != payload:
            raise AssertionError("Atomic JSON roundtrip failed")
        if len(read_csv(csv_path)) != len(rows):
            raise AssertionError("Atomic CSV roundtrip failed")


def test_local_paths(data_root: Path, resnet: Path, vit: Path) -> None:
    for path in (data_root / "train", data_root / "val", data_root / "devkit"):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path, minimum_bytes in ((resnet, 50_000_000), (vit, 200_000_000)):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size < minimum_bytes:
            raise AssertionError(f"Checkpoint is unexpectedly small: {path}")


def test_model_factories_and_matcher(device: torch.device) -> None:
    factories = {item["architecture_id"]: item for item in preflight.model_factories()}
    required = {"torchvision_resnet50", "torchvision_vit_b_16"}
    if not required.issubset(factories):
        raise AssertionError(f"Missing torchvision factories: {required - set(factories)}")

    for architecture_id in sorted(required):
        factory = factories[architecture_id]
        model = factory["factory"]()
        state = model.state_dict()
        prefixed = {f"module.{key}": value for key, value in state.items()}
        variants = dict(preflight.state_variants(prefixed))
        if "module." not in variants:
            raise AssertionError("module. prefix normalization is missing")
        match = preflight.match_state_to_model(
            variants["module."], model, factory["classifier_keys"]
        )
        if match["parameter_numel_coverage"] != 1.0:
            raise AssertionError(f"Incomplete parameter match for {architecture_id}")
        if not match["classifier_complete"]:
            raise AssertionError(f"Classifier did not match for {architecture_id}")
        model = model.to(device).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224, device=device))
        if tuple(output.shape) != (1, NUM_CLASSES) or not torch.isfinite(output).all():
            raise AssertionError(f"Invalid local factory output for {architecture_id}")
        del model, state, prefixed, variants
        if device.type == "cuda":
            torch.cuda.empty_cache()


def test_preprocessing() -> None:
    image = Image.new("RGB", (320, 256), color=(123, 45, 67))
    for interpolation in ("bilinear", "bicubic"):
        transform, profile = preflight.preprocessing_profile(
            {"interpolation": interpolation}
        )
        tensor = transform(image)
        if tuple(tensor.shape) != (3, 224, 224):
            raise AssertionError(f"Unexpected transformed shape: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise AssertionError("Preprocessing produced non-finite values")
        if profile["selection_rule"] != "architecture_default; not selected by accuracy":
            raise AssertionError("Preprocessing is not outcome-independent")


def test_role_gate() -> None:
    checkpoints = [
        {
            "checkpoint_id": "standard",
            "status": "PASS",
            "architecture": {"family": "resnet50"},
        },
        {
            "checkpoint_id": "anchor",
            "status": "PASS",
            "architecture": {"family": "vit_b16"},
        },
    ]
    assignment = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN",
        "standard_reference_checkpoint_id": "standard",
        "standard_reference_checkpoint_sha256": "standard-hash",
        "robust_anchor_checkpoint_id": "anchor",
        "robust_anchor_checkpoint_sha256": "anchor-hash",
        "robust_anchor_training_provenance": "Adversarially trained checkpoint with documented source.",
    }
    checkpoints[0]["sha256"] = "standard-hash"
    checkpoints[1]["sha256"] = "anchor-hash"
    status, blockers = preflight.validate_roles(assignment, checkpoints)
    if status != "BLOCKED" or not any("architecture-family mismatch" in item for item in blockers):
        raise AssertionError("Cross-family role assignment was not blocked")
    checkpoints[1]["architecture"]["family"] = "resnet50"
    status, blockers = preflight.validate_roles(assignment, checkpoints)
    if status != "PASS" or blockers:
        raise AssertionError(f"Valid same-family role assignment was blocked: {blockers}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    resnet = args.resnet50_checkpoint.expanduser().resolve()
    vit = args.vit_b16_checkpoint.expanduser().resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    test_offline_source_policy()
    print("PASS 1/7: sources contain no network/download execution path")
    test_split_contract()
    print("PASS 2/7: frozen five-way split is deterministic, disjoint, and balanced")
    test_atomic_roundtrip()
    print("PASS 3/7: atomic JSON/CSV contracts")
    test_local_paths(data_root, resnet, vit)
    print("PASS 4/7: local ImageNet directories and both checkpoint files exist")
    test_model_factories_and_matcher(device)
    print("PASS 5/7: local ResNet-50/ViT factories, 1000 logits, and key normalization")
    test_preprocessing()
    print("PASS 6/7: architecture-default 224x224 preprocessing")
    test_role_gate()
    print("PASS 7/7: cross-architecture or undocumented role assignment is blocked")
    print("ImageNet R0 self-test complete.")


if __name__ == "__main__":
    main()
