"""Original request-conditioned strengthening and Gaussian input noise."""

from __future__ import annotations

import torch
from torch import nn

from .config import DefenseConfig, positive_integer
from .metrics import ranked_topk


class CleanReferenceTopKDefense:
    """Return one FULL logit vector per image and defended request.

    standard_model defines the ONLINE reference on the current request input.
    That set is fixed during this request's strengthening steps and recomputed
    on the next request. It differs from the CLEAN set supplied to the attack.

    gradient_model and output_model are the normalized robust anchor in the
    paper. Models accept [0,1] pixels and normalize internally. Use frozen
    parameters and eval mode; this wrapper requests only input gradients.
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

    @property
    def sigma(self) -> float:
        """Stochasticity declaration expected by the original attack engine."""
        return self.config.sigma

    def objective(
        self, logits: torch.Tensor, reference_topk: torch.Tensor
    ) -> torch.Tensor:
        """Return w1 * winner margin + wk * exact-set boundary margin.

        The winner competes with EVERY other class, including selected ones.
        The set margin compares the weakest selected and strongest outside
        logits. Full class scores are required for both comparisons.
        """

        if logits.ndim != 2 or reference_topk.ndim != 2:
            raise ValueError("logits and reference_topk must both be rank two")
        if logits.shape[0] != reference_topk.shape[0]:
            raise ValueError("batch dimensions must agree")
        if reference_topk.shape[1] != self.config.top_k:
            raise ValueError("reference_topk width must equal config.top_k")

        if self.config.top_k >= logits.shape[1]:
            raise ValueError("top_k must be smaller than the number of classes")
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
            self.config.top1_weight * top1_margin + self.config.set_weight * set_margin
        )

    def reference_set(self, x: torch.Tensor) -> torch.Tensor:
        """Recompute the reference on this request, including attacked inputs."""
        with torch.no_grad():
            return ranked_topk(self.standard_model(x), self.config.top_k)

    def strengthen(self, x: torch.Tensor, reference_topk: torch.Tensor) -> torch.Tensor:
        """Original projected sign-gradient ascent in pixel coordinates.

        Projection is around the REQUEST input, not the attack's clean input.
        This does not guarantee monotone margins or certified robustness.
        """
        if self.config.steps and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "strengthening needs input gradients; use torch.no_grad(), "
                "not torch.inference_mode(), around defended requests"
            )

        request = x.detach()
        strengthened = request.clone()
        for _ in range(self.config.steps):
            with torch.enable_grad():
                variable = strengthened.detach().requires_grad_(True)
                score = self.objective(self.gradient_model(variable), reference_topk)
                gradient = torch.autograd.grad(score.sum(), variable, only_inputs=True)[
                    0
                ]
            strengthened = strengthened + self.config.alpha * gradient.sign()
            delta = (strengthened - request).clamp(
                -self.config.epsilon, self.config.epsilon
            )
            strengthened = (request + delta).clamp(0.0, 1.0).detach()
        return strengthened

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """One independent response with ALL class logits."""
        reference = self.reference_set(x)
        defended = self.strengthen(x, reference)
        if self.config.sigma > 0:
            defended = (
                defended + self.config.sigma * torch.randn_like(defended)
            ).clamp(0.0, 1.0)
        with torch.no_grad():
            return self.output_model(defended)

    def draw_logits(self, x: torch.Tensor, draws: int) -> torch.Tensor:
        """Independent responses shaped [draws,batch,classes].

        Reference, strengthening and fresh noise are recomputed per request.
        No clean-target or transformed-input cache is shared across requests.
        """

        positive_integer(draws, "draws")
        return torch.stack([self(x) for _ in range(draws)], dim=0)
