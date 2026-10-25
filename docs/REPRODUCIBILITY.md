# Reproduction and original experiment programs

The clean package provides the computational core and a tensor-evaluation CLI.
The archived programs provide the actual training, dataset manifests, stage
authorization, search, MC re-evaluation, finalizers and audits used by the
research. Do not infer formal reproduction from a successful CPU example.

## Required inputs

Datasets, checkpoint weights, parent-stage manifests and candidate tensors
are not included. The new CIFAR loader accepts the following original roles:

| Setting | Reference checkpoint | Anchor checkpoint |
|---|---|---|
| CIFAR-10 | `resnet20_cifar10.pth` | `resnet20_staged_epsilon_final_weights.pt` (robust branch) |
| CIFAR-100 | `resnet20_standard_best_validation_weights.pt` | `resnet20_teacher_top2_best_composite_weights.pt` |
| ImageNet / ResNet | Verified standard ResNet-50 | `Salman2020Do_R50`, trained at L-infinity 4/255 |
| ImageNet / ViT | Verified torchvision ViT-B/16 | The same robust ResNet-50 anchor |

Use original tensor fingerprints and file hashes from your execution artifacts.
Do not replace a missing checkpoint with freshly initialized or differently
trained weights and label its output as a reproduction.

CIFAR uses native 32×32 pixels. ImageNet uses shorter-side resize to 256,
center crop 224 and ToTensor: bilinear for the ResNet-reference pipeline,
bicubic for the ViT-reference pipeline. The same cropped pixels reach both
reference and anchor; normalization happens inside each model.

## Find a recorded entry point

Run from the repository root so sanitized relative data/model paths have the
documented meaning. Python adds the selected script's sibling directory to
its import path:

```bash
python -m pip install -e ".[experiments,test]"
python experiments/frozen/evaluate_topk_prior_baseline_attack_cifar10.py --help
python experiments/frozen/search_topk_cifar100_confirmatory_r5.py --help
python experiments/frozen/search_topk_imagenet_confirmatory_r5.py --help
python experiments/frozen/search_topk_imagenet_vit_reference_confirmatory_vr4.py --help
```

Review each script's path arguments and parent-artifact requirements before
running it. Some older bridges expect a declared external attack-core path;
supply the original dependency path where supported. The programs retain
historical assumptions rather than silently changing source-hash contracts.

## Stage order

| Family | Main stages |
|---|---|
| CIFAR-10 | teacher/anchor preparation → original holdout candidate search → MC1000 re-evaluation → paired aggregation |
| CIFAR-100 | R1 calibration → R2 model preparation → R3 clean evaluation → R4 disjoint attack pilot → R5 confirmatory |
| ImageNet / ResNet | R0 preflight → R1 anchor → R2 calibration → R3 clean → R4 disjoint pilot → R5 confirmatory |
| ImageNet / ViT | verified parent → VR1 calibration → VR2 clean → VR3 disjoint pilot → VR4 confirmatory |

Use stage self-tests, then prepare/search, reevaluate, finalize and audit.
Preserve original image identity, partitions, search seeds and checkpoint
fingerprints. Three fixed seeds are averaged per image before 50,000 paired
image bootstrap resamples. The two attack families each have a separate
5,000-response cap per image and seed.

CIFAR-10 selected one family globally per method on the same final holdout.
Do not describe that selection as a disjoint-pilot freeze. The other three
settings use the pilot-frozen family mapping in their JSON presets.

## Hash boundaries

No archived Python source was changed during v0.2.0 refactoring. The older
v0.1.0 package already sanitized machine-specific path defaults.
`experiments/frozen/SANITIZATION_MANIFEST.csv` records original and released
hashes. Some archived auditors require the exact original execution bytes
and paths; a sanitized or relocated copy can fail those historical checks.

The release manifest verifies this package. It does not rewrite or bypass
historical authorization checks and is not a substitute for the stage audit
associated with recorded results.

## New tensor evaluation

See README.md for `evaluate-tensors`. To export an already selected subset:

```python
torch.save(
    {"images": images.cpu(), "labels": labels.cpu().long(), "image_ids": image_ids},
    "datasets/selected_images.pt",
)
```

Selection, preprocessing and provenance must already be determined. The CLI
saves input/checkpoint hashes and scores each returned response separately.
Its batch seed labels follow the documented new runner; the archived pipeline
is authoritative for exact historical seed derivation and GPU behavior.

GPU training, original checkpoints and full dataset runs have not been
revalidated here. CPU checks cover refactor parity and synthetic checkpoints.
