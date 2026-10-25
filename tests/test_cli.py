"""Exercise installed entry points and the actual CIFAR loader/evaluation path."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import ROOT

from topk_fidelity.cli import main
from topk_fidelity.models.cifar_anchor import CIFARResNet20 as AnchorResNet20
from topk_fidelity.models.cifar_anchor import DualBackboneDefense
from topk_fidelity.models.cifar_reference import CIFARResNet20 as ReferenceResNet20


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_tensor_cli_loads_original_layout_and_writes_accounted_output(
    tmp_path, capsys, dataset
):
    config = json.loads((ROOT / "configs" / f"{dataset}.json").read_text())
    config["api_budget"] = 16
    config["reevaluation_draws"] = 3
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    torch.manual_seed(12)
    reference = (
        ReferenceResNet20("projection")
        if dataset == "cifar10"
        else AnchorResNet20(num_classes=100)
    )
    anchor = (
        DualBackboneDefense()
        if dataset == "cifar10"
        else AnchorResNet20(num_classes=100)
    )
    reference_path, anchor_path = tmp_path / "ref.pt", tmp_path / "anchor.pt"
    torch.save(reference.state_dict(), reference_path)
    torch.save(anchor.state_dict(), anchor_path)
    inputs = tmp_path / "images.pt"
    torch.save(
        {
            "images": torch.rand(1, 3, 32, 32),
            "labels": torch.tensor([0]),
            "image_ids": ["synthetic-case-0"],
        },
        inputs,
    )
    output = tmp_path / "output"
    result = main(
        [
            "evaluate-tensors",
            "--config",
            str(config_path),
            "--inputs",
            str(inputs),
            "--reference-checkpoint",
            str(reference_path),
            "--anchor-checkpoint",
            str(anchor_path),
            "--output-dir",
            str(output),
            "--method",
            "full",
            "--attack",
            "SquareAdaptive",
            "--seed",
            str(config["search_seeds"][0]),
        ]
    )
    assert result == 0
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["batches"][0]["api_queries"] == 16
    assert manifest["endpoint_evaluation_outside_attack_budget"] is True
    assert len(manifest["inputs_sha256"]) == 64
    assert (output / "per_image.csv").is_file()
    assert "examples" in capsys.readouterr().out


def test_inspect_cli_checks_json_without_loading_weights(capsys):
    assert main(["show-config", "--config", str(ROOT / "configs/cifar10.json")]) == 0
    assert json.loads(capsys.readouterr().out)["defense"]["top_k"] == 2
