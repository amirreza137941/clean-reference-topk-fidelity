# Reproducibility guide

## Scope

The public repository contains source code and small summaries. CIFAR/ImageNet
data, pretrained checkpoints, adversarial candidate tensors, and server output
trees are not redistributed. They must be obtained under their original terms.

## Expected local layout

Run experiment programs from `experiments/frozen/` unless a launcher states
otherwise. The sanitized defaults resolve relative to the working directory:

```text
repository/
├── datasets/
│   ├── cifar-10-batches-py/
│   ├── cifar-100-python/
│   └── imagenet/
├── models/
│   ├── resnet20/
│   ├── resnet50/
│   ├── vit/
│   └── imagenet_topk/
├── outputs/
└── experiments/frozen/
```

Several programs accept explicit `--data-*`, `--model-*`, or `--output-*`
arguments. Prefer those arguments on a cluster rather than editing source.

## Protocol invariants

1. The clean model prediction on the original input defines the reference set.
2. Exact fidelity compares unordered sets, not ranks.
3. Stochastic metrics are computed per returned response and then averaged.
4. Every constituent EOT/Monte Carlo draw consumes one API query.
5. Pilot data select configurations; confirmatory data estimate final effects.
6. The confirmatory analyses use three attack seeds and paired uncertainty.
7. Candidate generation and high-MC re-evaluation remain separate stages.

## Stage map

| Family | Stage | Role |
|---|---|---|
| CIFAR-100 | R1 | actual-defense calibration |
| CIFAR-100 | R2 | model preparation / dev Pareto |
| CIFAR-100 | R3 | official clean evaluation |
| CIFAR-100 | R4 | attack pilot |
| CIFAR-100 | R5 | confirmatory attack |
| CIFAR-100 | R6 | query-budget stress |
| CIFAR-100 | R7 | Gaussian-ensemble screen |
| ImageNet/ResNet | R0 | preflight |
| ImageNet/ResNet | R1 | robust anchor |
| ImageNet/ResNet | R2 | defense calibration |
| ImageNet/ResNet | R3 | clean confirmation |
| ImageNet/ResNet | R4 | attack pilot |
| ImageNet/ResNet | R5 | confirmatory attack |
| ImageNet/ResNet | posthoc | Q=20k, EOT=1 stress |
| ImageNet/ViT | VR1 | cross-architecture calibration |
| ImageNet/ViT | VR2 | clean confirmation |
| ImageNet/ViT | VR3 | attack pilot |
| ImageNet/ViT | VR4 | confirmatory attack (pending) |

## Safe order of operations

For a stage, use this order when the corresponding programs exist:

1. `test_*` or launcher self-test;
2. `prepare_*` / calibration;
3. search or candidate generation;
4. high-MC `reevaluate_*`;
5. `finalize_*`;
6. `audit_*`.

Do not report `--quick`, smoke-test, incomplete-shard, unaudited, or pilot-only
output as confirmatory evidence.

## Release-level checks

```bash
python -m compileall -q src experiments/frozen tests
python scripts/audit_release.py
pytest -q
```

The release audit checks identifying paths, likely credential strings, large
tracked files, and required repository files. It cannot prove that a new result
is scientifically valid; the stage-specific audit program remains authoritative.
