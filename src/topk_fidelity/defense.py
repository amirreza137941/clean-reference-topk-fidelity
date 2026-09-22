"""Clean-reference Top-k strengthening and stochastic output wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .metrics import ranked_topk


@dataclass(frozen=True)
class DefenseConfig:
    """Hyperparameters for the clean-reference wrapper."""

    top_k: int
    alpha: float
    epsilon: float
    steps: int
    sigma: float
    top1_weight: float = 1.0
    set_weight: float = 4.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.alpha < 0 or self.epsilon < 0 or self.sigma < 0:
            raise ValueError("alpha, epsilon, and sigma must be non-negative")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.top1_weight < 0 or self.set_weight < 0:
            raise ValueError("objective weights must be non-negative")


class CleanReferenceTopKDefense:
    """Strengthen a clean reference Top-k set, then optionally add noise.

    The standard model defines the reference set on the current request. A
    differentiable model supplies the strengthening gradient, and an output
    model returns the defended scores. These can be the same network or three
    different networks.
    """

    def __init__(
        self,
        standard_model: nn.Module,
        gradient_model: nn.Module,
        output_model: nn.Module,
        config: DefenseConfig,
    ) -> None:
        self.standard_model = standard_model
        self.gradient_model = gradient_model
        self.output_model = output_model
        self.config = config

    def objective(
        self, logits: torch.Tensor, reference_topk: torch.Tensor
    ) -> torch.Tensor:
        """Return the weighted Top-1 and set-boundary strengthening margin."""

        if logits.ndim != 2 or reference_topk.ndim != 2:
            raise ValueError("logits and reference_topk must both be rank two")
        if logits.shape[0] != reference_topk.shape[0]:
            raise ValueError("batch dimensions must agree")
        if reference_topk.shape[1] != self.config.top_k:
            raise ValueError("reference_topk width must equal config.top_k")

        selected = logits.gather(1, reference_topk)
        reference_top1 = reference_topk[:, :1]

        not_top1 = torch.ones_like(logits, dtype=torch.bool)
        not_top1.scatter_(1, reference_top1, False)
        strongest_not_top1 = logits.masked_fill(~not_top1, float("-inf")).max(1).values

        outside = torch.ones_like(logits, dtype=torch.bool)
        outside.scatter_(1, reference_topk, False)
        strongest_outside = logits.masked_fill(~outside, float("-inf")).max(1).values

        top1_margin = selected[:, 0] - strongest_not_top1
        set_margin = selected.min(1).values - strongest_outside
        return (
            self.config.top1_weight * top1_margin
            + self.config.set_weight * set_margin
        )

    def reference_set(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return ranked_topk(self.standard_model(x), self.config.top_k)

    def strengthen(
        self, x: torch.Tensor, reference_topk: torch.Tensor
    ) -> torch.Tensor:
        """Apply projected sign-gradient ascent inside an L-infinity ball."""

        clean = x.detach()
        strengthened = clean.clone()
        for _ in range(self.config.steps):
            with torch.enable_grad():
                variable = strengthened.detach().requires_grad_(True)
                score = self.objective(self.gradient_model(variable), reference_topk)
                gradient = torch.autograd.grad(score.sum(), variable, only_inputs=True)[0]
            strengthened = strengthened + self.config.alpha * gradient.sign()
            delta = (strengthened - clean).clamp(
                -self.config.epsilon, self.config.epsilon
            )
            strengthened = (clean + delta).clamp(0.0, 1.0).detach()
        return strengthened

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        reference = self.reference_set(x)
        defended = self.strengthen(x, reference)
        if self.config.sigma > 0:
            defended = (
                defended + self.config.sigma * torch.randn_like(defended)
            ).clamp(0.0, 1.0)
        with torch.no_grad():
            return self.output_model(defended)

    def draw_logits(self, x: torch.Tensor, draws: int) -> torch.Tensor:
        """Return independent defended responses with shape [draws,batch,class]."""

        if int(draws) < 1:
            raise ValueError("draws must be positive")
        return torch.stack([self(x) for _ in range(int(draws))], dim=0)
