"""Verify actual ResNet parameter layouts, loader routes and frozen settings."""

from __future__ import annotations

import ast

import pytest
import torch
from conftest import FROZEN, ROOT, original_symbols

from topk_fidelity.config import DefenseConfig, ExperimentConfig
from topk_fidelity.models import load_cifar10_models, load_cifar100_models
from topk_fidelity.models.cifar_anchor import DualBackboneDefense
from topk_fidelity.models.cifar_reference import CIFARResNet20
from topk_fidelity.models.normalization import NormalizedModel, RobustPixelModel


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("block_type", ["projection", "option_a"])
def test_cifar10_reference_and_dual_checkpoint_routes(tmp_path, wrapped, block_type):
    original_reference = original_symbols(
        "evaluate_top1_set2_gated_cifar10.py",
        [
            "LambdaLayer",
            "OptionABasicBlock",
            "ProjectionBasicBlock",
            "CIFARResNet20",
            "NormalizedModel",
        ],
    )
    original_dual = original_symbols(
        "train_resnet20_dual_backbone_top1_set2_cifar10.py",
        ["ProjectionBasicBlock", "CIFARResNet20", "DualBackboneDefense"],
    )
    torch.manual_seed(9)
    old_reference = original_reference.NormalizedModel(
        original_reference.CIFARResNet20(block_type),
        std=(0.2471, 0.2435, 0.2616),
    ).eval()
    old_dual = original_dual.DualBackboneDefense().eval()
    standard_path, dual_path = tmp_path / "reference.pt", tmp_path / "dual.pt"
    state = old_reference.state_dict() if wrapped else old_reference.model.state_dict()
    torch.save(
        {"state_dict": {"module." + key: value for key, value in state.items()}},
        standard_path,
    )
    torch.save({"model_state_dict": old_dual.state_dict()}, dual_path)
    new_reference, new_anchor = load_cifar10_models(standard_path, dual_path)
    images = torch.rand(2, 3, 32, 32, requires_grad=True)
    expected_reference = old_reference(images)
    expected_anchor = old_dual.robust_branch(old_dual.normalize(images))
    torch.testing.assert_close(
        new_reference(images), expected_reference, rtol=0, atol=0
    )
    torch.testing.assert_close(new_anchor(images), expected_anchor, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected_anchor.sum(), images)[0]
    new_grad = torch.autograd.grad(new_anchor(images).sum(), images)[0]
    torch.testing.assert_close(new_grad, expected_grad, rtol=0, atol=0)
    assert all(not p.requires_grad for p in new_anchor.parameters())
    assert not torch.equal(new_anchor(images), old_dual(images))


@pytest.mark.parametrize("wrapped", [False, True])
def test_cifar100_loader_preserves_original_fixed_normalization(tmp_path, wrapped):
    original = original_symbols(
        "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
        ["ProjectionBasicBlock", "CIFARResNet20", "NormalizedModel"],
    )
    torch.manual_seed(19)
    old = original.NormalizedModel(original.CIFARResNet20(num_classes=100)).eval()
    state = old.state_dict() if wrapped else old.model.state_dict()
    # Accept the same shortcut/linear aliases as the original R2 loader.
    state = {
        key.replace(".downsample.", ".shortcut.").replace("fc.", "linear."): value
        for key, value in state.items()
    }
    path = tmp_path / "weights.pt"
    torch.save({"net": state}, path)
    reference, anchor = load_cifar100_models(path, path)
    images = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(reference(images), old(images), rtol=0, atol=0)
    torch.testing.assert_close(anchor(images), old(images), rtol=0, atol=0)


def test_cifar10_stored_reference_normalization_is_preserved(tmp_path):
    model = NormalizedModel(
        CIFARResNet20("projection"), (0.4, 0.5, 0.6), (0.1, 0.2, 0.3)
    )
    reference_path, dual_path = tmp_path / "reference.pt", tmp_path / "dual.pt"
    torch.save(model.state_dict(), reference_path)
    torch.save(DualBackboneDefense().state_dict(), dual_path)
    reference, _ = load_cifar10_models(reference_path, dual_path)
    torch.testing.assert_close(reference.mean, model.mean, rtol=0, atol=0)
    torch.testing.assert_close(reference.std, model.std, rtol=0, atol=0)


def _constant(filename, name):
    tree = ast.parse((FROZEN / filename).read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )
    return ast.literal_eval(node.value)


@pytest.mark.parametrize(
    "filename,source,key",
    [
        (
            "cifar100.json",
            "topk_cifar100_attack_pilot_r4_common.py",
            "EXPECTED_FROZEN_DEFENSE",
        ),
        (
            "imagenet_resnet.json",
            "topk_imagenet_attack_pilot_r4_common.py",
            "FROZEN_DEFENSE",
        ),
        (
            "imagenet_vit.json",
            "topk_imagenet_vit_reference_attack_pilot_vr3_common.py",
            "FROZEN_DEFENSE",
        ),
    ],
)
def test_json_presets_match_original_frozen_defense(filename, source, key):
    config = ExperimentConfig.from_file(ROOT / "configs" / filename)
    old = _constant(source, key)
    assert config.defense.alpha == old["alpha_px"] / 255
    assert config.defense.epsilon == old["epsilon_px"] / 255
    assert config.defense.steps == old["steps"]
    assert config.defense.sigma == old["sigma"]
    assert config.defense.top1_weight == old["top1_weight"]
    assert config.defense.set_weight == old["set_weight"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1])
def test_defense_config_rejects_nonfinite_and_negative_noise(bad):
    with pytest.raises(ValueError):
        DefenseConfig(top_k=2, alpha=0.005, epsilon=0.03, steps=2, sigma=bad)


def test_cifar10_deterministic_refresh_is_explicit():
    config = ExperimentConfig.from_file(ROOT / "configs" / "cifar10.json")
    assert config.adaptive_for(False).refresh_eot == 0
    assert config.adaptive_for(True).refresh_eot == 1
    assert config.mean_logit_reduction == "sequential_sum"


def test_robust_adapter_calls_only_the_normalized_anchor():
    dual = DualBackboneDefense().eval()
    images = torch.rand(1, 3, 32, 32)
    torch.testing.assert_close(
        RobustPixelModel(dual)(images),
        dual.robust_branch(dual.normalize(images)),
        rtol=0,
        atol=0,
    )
