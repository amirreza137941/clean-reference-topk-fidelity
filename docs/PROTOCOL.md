# Computational contract

## Defense

For the current request u, the reference model supplies ranked indices R(u)
and a winner r1. R(u) is held fixed for T input updates and recomputed on the
next API request. With anchor logits z, the objective is

```text
winner_margin = z[r1] - max(z[j] for j != r1)
set_margin = min(z[j] for j in R(u)) - max(z[j] for j outside R(u))
objective = top1_weight * winner_margin + set_weight * set_margin
```

Each step ascends the sign of the input gradient, projects around u in an
L-infinity ball and clips to [0,1]. Fresh Gaussian input noise is added after
strengthening and clipped. The output model returns the full C-dimensional
logit vector. No logit truncation, argmax surrogate, parameter optimization,
cached clean reference or monotone-margin guarantee is added.

| Setting | k | T | Step alpha | Internal radius | Sigma | Attack radius |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 2 | 2 | 0.005 | 0.03 | 0.02144277648925781 | 2/255 |
| CIFAR-100 | 2 | 2 | 1.275/255 | 8/255 | 0.015 | 2/255 |
| ImageNet, either reference | 5 | 4 | 1/255 | 4/255 | 0.025 | 4/255 |

Weights are 1 and 4. Robust disables strengthening and noise; Full-sigma0
disables noise; Noise-only disables strengthening; Full enables both.
CIFAR-10 Full-sigma0 has no comparable confirmatory row and no selected family
in its preset.

## Attack API

```python
from topk_fidelity.attacks import run_square_batch
from topk_fidelity.metrics import ranked_topk

# clean_reference_logits are computed once from CLEAN images, outside search.
candidate, details = run_square_batch(
    attack="SquareAdaptive",
    defense=defense,
    clean=images,
    reference_topk=ranked_topk(clean_reference_logits, settings.defense.top_k),
    eps=settings.attack_epsilon,
    api_budget=settings.api_budget,
    p_init=settings.p_init,
    adaptive_config=settings.adaptive_for(defense.sigma > 0).as_dict(),
    racing_config=settings.racing.as_dict(),
    proposal_seed=proposal_seed,
    oracle_seed=oracle_seed,
    mean_logit_reduction=settings.mean_logit_reduction,
)
```

The attack explicitly receives the clean target set and full defended scores.
It does not receive model parameters or gradients. Internal defense forwards
and input gradients do not add extra API responses to the attack ledger.

Stochastic Adaptive uses 1 search draw, 1 refresh draw, 5 confirmation
candidates and a 20% nominal confirmation reservation. Stochastic Racing uses
5/5 search/refresh draws, an 8-candidate archive, a race every 25 iterations,
5 draws per racing member, temperature 0.1 and a 25% nominal confirmation
reservation. Exact integer plans allocate leftover queries to confirmation.

Racing's deterministic specialization uses one search draw, zero refresh,
one confirmation candidate and no periodic races. Deterministic Adaptive
retains its 5-candidate pool; refresh is 0 for CIFAR-10 and 1 for the R4 route.
The original proposal distribution, sign-attempt order and patch-size schedule
are retained.

A ledger covers a fixed batch. Every batch member has the same charged count.
Do not reuse one ledger for unrelated image batches, families or search seeds.

## Independent endpoint and interpretation

Endpoint MC evaluation is performed after the confirmed candidate is fixed.
Its draws are outside search Q and cannot be used for selecting another
candidate, seed or per-image attack family.

ExactSet, reference-winner agreement and ground-truth accuracy are separate
events. Hidden failure is the same-response event
reference-winner agreement AND exact-set failure. It is not a product of two
independently averaged probabilities.

Full logits remain useful for score-based attacks and for applications that
use alternatives and score magnitudes. The code does not claim that ExactSet
preserves confidence calibration, the remaining logits, or downstream
clinical decisions.

For the paper's confidence intervals, image identities are aligned and the
three fixed seed results are averaged before paired image resampling.
Neither independent seed bootstrapping nor per-image family envelopes should
replace the archived finalizers. The CIFAR-10 family-selection uncertainty
is not included in its conditional bootstrap interval.

## Gradient context

The defense temporarily enables autograd for the input. Outer
`torch.no_grad()` is supported. PyTorch's
[inference-mode documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
explains why inference tensors are unsuitable for autograd-tracked work.
