# Clean-Reference Top-k Fidelity

Readable, reusable code for the original request-conditioned Top-k defense and
its score-based SquareAdaptive / SquareRacing evaluation. The refactor keeps
the original margin objective, input updates, noise, proposal RNG order,
candidate archives and query allocations. Existing experiment programs remain
available with their source identities.

**English** · [راهنمای فارسی](README_FA.md) · [Migration map](docs/MIGRATION.md)

## Start here

Install a PyTorch build for your CPU/CUDA platform, then install the package
from the repository root:

```bash
python -m pip install -e ".[test,dev]"
python -m topk_fidelity demo --attack SquareAdaptive
python -m topk_fidelity demo --attack SquareRacing
python -m topk_fidelity show-config --config configs/cifar10.json
```

The CPU examples use tiny synthetic models and images. Their numbers are
demonstration output and are not manuscript results. For the historical
dataset-backed programs, install `".[experiments]"` as well. Supported package
requirements are in [pyproject.toml](pyproject.toml); the locally tested
environment is recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

## Code map

| Path | Responsibility |
|---|---|
| `src/topk_fidelity/defense.py` | Original online reference, strengthening and full-logit response |
| `src/topk_fidelity/config.py` | Validated parameters and method ablations |
| `src/topk_fidelity/attacks/` | Original objectives, query plans, proposals, archives and search loops |
| `src/topk_fidelity/query.py` | Per-image response accounting and both mean-logit reduction routes |
| `src/topk_fidelity/metrics.py`, `evaluation.py` | Unordered-set metrics and independent per-response endpoint evaluation |
| `src/topk_fidelity/models/` | Original CIFAR architectures, normalization and checkpoint loaders |
| `src/topk_fidelity/training/losses.py` | Original teacher Top-2 ranking and distillation losses |
| `configs/` | Explicit CIFAR-10, CIFAR-100, ImageNet/ResNet and ImageNet/ViT settings |
| `experiments/frozen/` | Recorded training, preparation, search, finalization and audit programs |
| `tests/` | CPU comparisons against the actual archived source |
| `results/` | Small summaries from the revised manuscript; no new experiments |

The old imports from `topk_fidelity.defense` and `topk_fidelity.attacks.square`
remain usable. [docs/refactor_provenance.json](docs/refactor_provenance.json)
maps moved symbols to source files, line spans and hashes.

## Use with your models

Both models receive pixels in `[0,1]` and normalize internally. Freeze model
parameters and use evaluation mode; the defense still needs input gradients.

```python
from topk_fidelity import (
    CleanReferenceTopKDefense,
    ExperimentConfig,
    evaluate_responses,
)
from topk_fidelity.models import load_cifar10_models

settings = ExperimentConfig.from_file("configs/cifar10.json")
reference, anchor = load_cifar10_models(
    "checkpoints/resnet20_cifar10.pth",
    "checkpoints/resnet20_staged_epsilon_final_weights.pt",
    device="cpu",
)
defense = CleanReferenceTopKDefense(
    standard_model=reference,
    gradient_model=anchor,
    output_model=anchor,
    config=settings.defense.for_method("full"),
)
logits = defense(images)  # [batch, ALL classes]
```

`load_cifar10_models` uses only the normalized `robust_branch` of the dual
checkpoint. `load_cifar100_models` loads the original standard and teacher-Top2
anchor weights. For ImageNet, pass the already verified, normalized reference
and ResNet-50 anchor to the same defense API. In the ViT setting, ViT-B/16 is
the **online reference**; the gradient and output network is ResNet-50.

## Evaluate existing CIFAR tensors

A tensor file must contain float32 `images` shaped `[N,3,32,32]`, int64 `labels`,
and unique `image_ids`. The command records checkpoint/input hashes, effective
parameters and per-image query counts:

```bash
python -m topk_fidelity evaluate-tensors \
  --config configs/cifar10.json \
  --inputs datasets/selected_images.pt \
  --reference-checkpoint checkpoints/resnet20_cifar10.pth \
  --anchor-checkpoint checkpoints/resnet20_staged_epsilon_final_weights.pt \
  --method full --attack SquareAdaptive --seed 11071 \
  --device cpu --batch-size 4 --output-dir outputs/cifar10_full
```

The command writes `per_image.csv` and `run.json` to an empty output directory.
It evaluates the supplied tensors; it does not infer the paper's frozen split,
pilot authorization, or historical seed labels for every dataset. Use the
[original stage guide](docs/REPRODUCIBILITY.md) to replay manuscript protocols.
The new CLI and original pipeline are separate entry points.

## Protocol details that matter

- The evaluator/attacker's target is the **clean** reference set `S_k(x)`.
  The defense recomputes its **online** set on each requested input and holds
  that set fixed only within that request's strengthening steps.
- Strengthening performs projected sign-gradient **ascent** on
  `1 × winner margin + 4 × set-boundary margin`. Its radius is separate
  from the attack radius.
- Noise is added after strengthening, followed by pixel clipping. Each response
  returns **all class logits**. Top-k evaluation does not establish preservation
  of the other logits, their magnitudes, calibration or medical utility.
- ExactSet compares unordered sets. Ground-truth Top-1 accuracy, reference
  winner agreement, ground-truth coverage and set overlap are different metrics.
- Primary stochastic metrics score **each response before averaging**.
  SquareAdaptive minimizes the margin of mean logits. SquareRacing minimizes
  the mean per-draw sigmoid margin. Those objectives remain distinct.
- Initialization, EOT, refresh, races and in-attack confirmation all consume the
  5,000-response budget **per image, family and seed**. Independent endpoint
  MC evaluation uses 1,000 draws for stochastic methods, outside that budget,
  and must not select a new candidate.
- CIFAR-10's deterministic Adaptive path has zero refresh and sequential logit
  accumulation. The generic CIFAR-100/ImageNet route retains its original
  refresh setting and stacked mean. Both are explicit in the JSON settings.
- CIFAR-100 and ImageNet family choices were frozen using disjoint pilots.
  CIFAR-10 chose a global family per method on the same final holdout; that
  selection limitation remains part of the interpretation.

See [docs/PROTOCOL.md](docs/PROTOCOL.md) for parameters, attack use and limits.
Use `torch.no_grad()` for outer evaluation contexts; strengthening cannot run
inside `torch.inference_mode()` because it requires input gradients.

## Check the refactor

```bash
ruff check .
ruff format --check .
python -m compileall -q src tests examples experiments/frozen
pytest -q
python scripts/audit_release.py
sha256sum -c MANIFEST.sha256
```

Tests compare original and refactored logits, objective gradients, actual
ResNet checkpoint routes, attack candidates and query ledgers on CPU. They
also exercise the tensor CLI. GPU training and full dataset experiments have
not been repeated for this refactor. No speedup or new robustness claim is made.

## Results and release metadata

[results/paper_summary.csv](results/paper_summary.csv) reflects the revised
Pattern Analysis and Applications manuscript, including the completed
ViT-reference summary. These are existing experiment summaries, not outputs
of the CPU examples. The original weights and datasets are not in this archive.

The existing [MIT license](LICENSE) and
[Square Attack BSD notice](THIRD_PARTY_NOTICES.md) are retained.
`CITATION.cff.template` is an unfinished metadata template; fill the real
authors and repository URL before publishing a release. No repository URL,
DOI or publication status is fabricated.
