# From the original scripts to reusable modules

The repository preserves the supplied experiment snapshot and extracts its
repeated computational core into a documented package. It does not replace
the experiment algorithms with placeholder attacks or different networks.

| Original definition | New location |
|---|---|
| `TargetedGaussianDefense`, `TopKTargetedGaussianDefense` | `defense.py::CleanReferenceTopKDefense` |
| `square_reference_p_selection`, query plans | `attacks/plans.py` |
| `exact_set_attack_margin`, per-draw survival score | `attacks/objectives.py` |
| Vertical stripes and Square proposals | `attacks/proposals.py` |
| Accepted-candidate / racing archives | `attacks/archive.py` |
| CIFAR-10's generic score-function Adaptive loop | `attacks/adaptive.py` |
| CIFAR-10's generic score-function Racing loop | `attacks/racing.py` |
| Dataset-agnostic R4 `run_square_batch` interface | `attacks/runner.py` |
| `VariableQueryOracle` and `PerDrawQueryOracle` | `query.py` |
| Original CIFAR-10 standard-model blocks | `models/cifar_reference.py` |
| Original projection backbone and dual container | `models/cifar_anchor.py` |
| `RobustPixelModel` and normalization | `models/normalization.py` |
| CIFAR-10 / CIFAR-100 checkpoint loading routes | `models/checkpoints.py` |
| Teacher Top-2 ranking and distillation | `training/losses.py` |

Paths in the right column are relative to `src/topk_fidelity/`.
[refactor_provenance.json](refactor_provenance.json) gives original source
hashes and line spans. Type hints and formatting may change; computational
operations are retained and differential tests load the actual archived
definitions.

## Details kept deliberately

The CIFAR standard/reference backbone uses `shortcut` parameter names; the
anchor uses `downsample`. They remain separate definitions to preserve the
original parameter layouts and forward/ReLU behavior. The checkpoint loader
retains the original key aliases. Stored CIFAR-10 reference mean/std buffers
remain effective. CIFAR-100 uses the fixed R2 normalization, as its original
loader did.

The CIFAR-10 dual checkpoint remains a complete checkpoint container. Its
composed output is not passed to the paper's Top-k wrapper. The
`RobustPixelModel` adapter selects only the normalized robust branch.

CIFAR-10 Adaptive uses a zero-refresh deterministic path and sequential
floating-point logit summation. The generic R4 path uses its supplied refresh
setting and stacked mean. The new oracle supports both; JSON config and
`run_square_batch(mean_logit_reduction=...)` select the route explicitly.

The generic scoring-function loops from CIFAR-10 also implement the R4
exact-set search when given R4 parameters. Differential tests compare their
full returned candidates, allocation details and counters with R4, including
stochastic racing rounds.

## Intentional interface changes

- Invalid/nonfinite defense parameters now fail before execution.
- A hard ledger limit prevents a new oracle call from exceeding the budget.
- Strengthening under `torch.inference_mode()` gives a clear error; use
  `torch.no_grad()` so its local input-gradient context can work.
- New CIFAR loaders read tensor state dictionaries with
  `weights_only=True`; serialized executable model objects are not supported.
  Architecture, tensor loading routes and normalization are compared using
  synthetic checkpoints in the original layouts.
- `evaluate-tensors` accepts explicit files and writes into an empty output
  directory, avoiding accidental replacement of previous run outputs.
- Endpoint evaluation accumulates per-response metrics in float64. Chunk size
  changes memory use only; it does not average logits before scoring.

## Boundaries

Training schedules, PGD generation, checkpoint selection, dataset splitting,
baseline implementations (AAA/RFD), launcher recovery, official bootstrap
finalization and historical hash checks stay in `experiments/frozen/`.
Their Python sources are unchanged relative to the supplied v0.1.0 ZIP.

The new tensor CLI is for supplied CIFAR tensors and existing weights. Its
saved metadata explicitly says that original split/audit status is not
inferred. The historical ImageNet model-verification and preprocessing
pipeline is retained; ImageNet networks can also be passed to the clean API.
