"""Load existing CIFAR checkpoints with their original normalization routes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from .cifar_anchor import CIFARResNet20 as AnchorResNet20
from .cifar_anchor import DualBackboneDefense
from .cifar_reference import CIFARResNet20 as ReferenceResNet20
from .normalization import NormalizedModel, RobustPixelModel, freeze_model

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def read_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a state dictionary or the original model/state_dict containers."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")
    for name in ("model_state_dict", "state_dict", "model", "net"):
        if isinstance(payload.get(name), Mapping) and payload[name]:
            payload = payload[name]
            break
    state = dict(payload)
    if not state or not all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
    ):
        raise ValueError("checkpoint state must have string keys and tensor values")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state


def _load_matching(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    allow_normalization: bool = False,
) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if allow_normalization:
        missing = [key for key in missing if key not in {"mean", "std"}]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
        )


def load_cifar10_models(
    reference_weights: str | Path,
    dual_weights: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, nn.Module]:
    """Load the original standard teacher and the dual checkpoint's anchor.

    Wrapped reference checkpoints retain their stored mean/std. Bare references
    use the notebook normalization. The dual's composed logits are not used.
    """
    state = {
        key.replace(".downsample.", ".shortcut."): value
        for key, value in read_state_dict(reference_weights).items()
    }
    wrapped = "model.conv1.weight" in state
    body_state = (
        {
            key[len("model.") :]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
        if wrapped
        else state
    )
    block_type = (
        "projection" if "layer2.0.shortcut.0.weight" in body_state else "option_a"
    )
    body = ReferenceResNet20(block_type=block_type, num_classes=10)
    reference = NormalizedModel(body, CIFAR10_MEAN, CIFAR10_STD)
    if wrapped:
        _load_matching(reference, state, allow_normalization=True)
    else:
        _load_matching(body, body_state)
    dual = DualBackboneDefense(selector_temperature=1.0, primary_margin=1.0)
    _load_matching(dual, read_state_dict(dual_weights))
    return (
        freeze_model(reference.to(device)),
        freeze_model(RobustPixelModel(dual).to(device)),
    )


def _load_cifar100_model(path: str | Path, device: str | torch.device) -> nn.Module:
    state = read_state_dict(path)
    if "model.conv1.weight" in state:
        state = {
            key[len("model.") :]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
    state = {
        (
            "fc." + key[len("linear.") :]
            if key.startswith("linear.")
            else key.replace(".shortcut.", ".downsample.")
        ): value
        for key, value in state.items()
    }
    body = AnchorResNet20(num_classes=100)
    _load_matching(body, state)
    # R2's original CIFAR-100 loader deliberately uses these fixed buffers.
    return freeze_model(NormalizedModel(body, CIFAR100_MEAN, CIFAR100_STD).to(device))


def load_cifar100_models(
    reference_weights: str | Path,
    anchor_weights: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, nn.Module]:
    return (
        _load_cifar100_model(reference_weights, device),
        _load_cifar100_model(anchor_weights, device),
    )
