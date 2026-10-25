#!/usr/bin/env python3
"""Pre-download self-test for ImageNet robust-anchor R1."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch import nn

import prepare_topk_imagenet_robust_anchor_r1 as stage
from topk_imagenet_robust_anchor_common_r1 import (
    FROZEN_SPLIT_SHA256,
    PARENT_PROTOCOL_ID,
    PROTOCOL_ID,
    ROBUSTBENCH_EPSILON,
    ROBUSTBENCH_GDRIVE_ID,
    ROBUSTBENCH_MODEL_ID,
    STANDARD_CHECKPOINT_SHA256,
    source_provenance,
    validate_download_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test ImageNet robust-anchor R1")
    parser.add_argument("--project-dir", type=Path, default=stage.DEFAULT_PROJECT_DIR)
    parser.add_argument("--r0-dir", type=Path, default=stage.DEFAULT_R0_DIR)
    parser.add_argument(
        "--standard-checkpoint", type=Path, default=stage.DEFAULT_STANDARD_CHECKPOINT
    )
    return parser.parse_args()


def test_frozen_source_contract() -> None:
    source = source_provenance()
    assert PROTOCOL_ID == "imagenet_topk_robust_anchor_r1_20260910"
    assert PARENT_PROTOCOL_ID == "imagenet_topk_preflight_r0_20260910"
    assert len(FROZEN_SPLIT_SHA256) == 64
    assert len(STANDARD_CHECKPOINT_SHA256) == 64
    assert ROBUSTBENCH_MODEL_ID == "Salman2020Do_R50"
    assert ROBUSTBENCH_GDRIVE_ID == "1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r"
    assert source["architecture"] == "ResNet-50"
    assert source["preprocessing"] == "Res256Crop224"
    assert source["epsilon_linf_255"] == 4.0
    print("PASS 1/7: official source/model/threat/preprocessing locks")


def test_source_scope(project_dir: Path) -> None:
    names = [
        "topk_imagenet_robust_anchor_common_r1.py",
        "prepare_topk_imagenet_robust_anchor_r1.py",
        "audit_topk_imagenet_robust_anchor_r1.py",
        "run_topk_imagenet_robust_anchor_r1.py",
    ]
    joined = "\n".join((project_dir / name).read_text(encoding="utf-8") for name in names)
    forbidden = [
        "pip install",
        "torch.hub.load",
        "download_and_extract_archive",
        "download_url(",
        "weights_only=False",
        "train()",
        "optimizer.step",
    ]
    for token in forbidden:
        assert token not in joined, f"Forbidden source token: {token}"
    assert joined.count(ROBUSTBENCH_GDRIVE_ID) >= 1
    assert "ImageNet is not downloaded" in joined
    print("PASS 2/7: no package install, dataset download, training, or unsafe torch.load")


def test_download_guards() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        html = root / "response.pt"
        html.write_bytes(b"<!doctype html><html>access denied</html>")
        try:
            validate_download_candidate(html)
        except ValueError:
            pass
        else:
            raise AssertionError("Small HTML response passed checkpoint guard")
        absent = root / "absent.pt"
        try:
            validate_download_candidate(absent)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Absent checkpoint passed checkpoint guard")
    print("PASS 3/7: failed/HTML download guards")


def test_pgd_row_selection() -> None:
    rows = [
        {"partition": "preflight_sanity", "relative_path": f"val/{index}.JPEG", "target": index}
        for index in range(25)
    ]
    left = stage.select_pgd_rows(rows, 10)
    right = stage.select_pgd_rows(list(reversed(rows)), 10)
    assert left == right
    assert len(left) == 10
    assert len({row["relative_path"] for row in left}) == 10
    print("PASS 4/7: outcome-blind deterministic PGD subset")


def test_pgd_projection() -> None:
    torch.manual_seed(123)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 5)).eval()
    model.requires_grad_(False)
    images = torch.rand(3, 3, 4, 4)
    targets = torch.tensor([0, 1, 2])
    adversarial_a = stage.pgd_linf(
        model, images, targets, ROBUSTBENCH_EPSILON, 1.0 / 255.0, 3, 99
    )
    adversarial_b = stage.pgd_linf(
        model, images, targets, ROBUSTBENCH_EPSILON, 1.0 / 255.0, 3, 99
    )
    assert torch.equal(adversarial_a, adversarial_b)
    assert float((adversarial_a - images).abs().max()) <= ROBUSTBENCH_EPSILON + 1e-7
    assert float(adversarial_a.min()) >= 0.0
    assert float(adversarial_a.max()) <= 1.0
    print("PASS 5/7: deterministic feasible PGD identity diagnostic")


def test_preprocessing() -> None:
    _, report = stage.raw_preprocessing()
    assert report["resize_shorter_side"] == 256
    assert report["crop_size"] == 224
    assert report["interpolation"] == "bilinear"
    assert report["normalization_inside_model"] is True
    print("PASS 6/7: Res256Crop224 and in-model normalization contract")


def test_live_parent(r0_dir: Path, standard_checkpoint: Path) -> None:
    parent = stage.load_and_validate_parent(
        r0_dir.expanduser().resolve(), standard_checkpoint.expanduser().resolve()
    )
    assert len(parent["sanity_rows"]) == 1000
    assert parent["split_manifest"]["split_sha256"] == FROZEN_SPLIT_SHA256
    assert parent["standard_report"]["sha256"] == STANDARD_CHECKPOINT_SHA256
    print("PASS 7/7: live R0 split and local standard ResNet-50 are unchanged")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    test_frozen_source_contract()
    test_source_scope(project_dir)
    test_download_guards()
    test_pgd_row_selection()
    test_pgd_projection()
    test_preprocessing()
    test_live_parent(args.r0_dir, args.standard_checkpoint)
    print("ImageNet robust-anchor R1 self-test complete.")


if __name__ == "__main__":
    main()
