# Contributing

Keep changes focused and explain any change to scientific behavior.

For core changes, run `ruff check .`, `ruff format --check .`, `pytest -q`
and `python scripts/audit_release.py`. The differential tests use the actual
archived definitions, so they are part of the review of numerical changes.

Preserve frozen seeds, splits, budgets, checkpoint hashes, MC counts and
family-selection rules. A protocol change needs a new stage/version rather
than overwriting recorded experiment programs. Update the migration notes
when moving symbols, and regenerate the release manifest for a new release.
Use `python scripts/update_manifest.py` after the final edits.

Avoid generated outputs, datasets and checkpoints in commits. Root data
directories are ignored; source modules under `src/topk_fidelity/models/`
are intentionally tracked.
