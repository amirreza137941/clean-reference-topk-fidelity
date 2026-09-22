# Frozen experiment programs

This directory is a flat, sanitized snapshot of the experiment code. It stays
flat because the historical launchers import sibling modules directly and
verify stage dependencies by filename and, in some cases, by source hash.

## What was changed for release

- machine-specific home directories were replaced by repository-relative
  defaults;
- no result tensors, datasets, checkpoints, lock files, or server logs were
  included;
- filenames and protocol stage identifiers were preserved.

`SANITIZATION_MANIFEST.csv` records the SHA-256 digest of every original source
and its released counterpart. Only 48 of 167 programs required path
sanitization; the remaining files are byte-identical to their source snapshot.

No scientific hyperparameter was intentionally changed by sanitization.

## Before running

1. Read `../../docs/REPRODUCIBILITY.md`.
2. Install the experiment dependencies with
   `python -m pip install -e ".[experiments,test]"` from the repository root.
3. Supply dataset, checkpoint, and output paths through command-line arguments
   whenever available.
4. Run the relevant `test_*` program before a launcher.
5. Keep pilot and confirmatory outputs in separate directories.

Some stage-level audit programs validate hashes from the original execution
artifact. A sanitized file can therefore differ from the old embedded source
hash even though only its default path changed. Use `SANITIZATION_MANIFEST.csv`
to bridge the original and public digests. The public release audit establishes
integrity of this snapshot; scientific results remain tied to the original
recorded artifact hashes.

## Navigation by filename

- `evaluate_*`, `calibrate_*`, `prepare_*`: model/data preparation or direct
  evaluation;
- `search_*`: adversarial candidate search;
- `reevaluate_*`: independent high-MC candidate evaluation;
- `finalize_*`: paired aggregation and report creation;
- `audit_*`: protocol and artifact checks;
- `run_*`, `*_launcher.py`: orchestration;
- `test_*`: stage-level self-tests;
- `*_common.py`, `*_core_*.py`: shared definitions and attack primitives.

The current ViT cross-architecture confirmatory family ends in `VR4`. Its
result is pending until the corresponding finalizer and auditor pass.
