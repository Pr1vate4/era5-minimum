# ERA5-Minimum: repository instructions for Codex

## Mission
Build a reproducible research system that estimates the minimum representative ERA5 training sample required for a 32x–64x neural autoencoder while preserving scientifically meaningful reconstruction quality.

## Non-negotiable rules
- Do not claim a binary compression ratio when only tensor element counts were measured.
- Keep tensor compression ratio and real serialized compression ratio as separate metrics.
- Never change metric definitions, data splits, normalization, channel ordering, or seeds silently.
- Prevent temporal leakage: validation/test timestamps must not overlap or be adjacent copies of training samples.
- Fit normalization statistics on the training split only.
- Report per-variable metrics in physical units after inverse normalization.
- Use latitude-weighted spatial metrics for global grids.
- Every experiment must save config, git commit when available, seed, metrics, and runtime.
- Prefer a small working baseline over an unverified complex architecture.
- Do not add a dependency unless it is necessary and documented.
- Never commit ERA5 credentials, `.cdsapirc`, large datasets, model checkpoints, or generated outputs.

## Project conventions
- Official team and CI version: Python 3.12; the canonical dependency source is `pyproject.toml`.
- Source code lives in `src/era5_minimum`.
- Config-driven experiments live in `configs/`.
- Tests live in `tests/`.
- [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) is the current source of truth for ERA5 schema, canonical channels, units, masks, and precipitation semantics.
- ERA5 daily downloads use `scripts/download_era5.py`; `scripts/download_era5_range.py` may only orchestrate it. Both dry-runs must remain offline and must not import `cdsapi`.
- CDS credentials belong only in `~/.cdsapirc`; never read, print, copy, or place them in project artifacts.
- Research decisions go to `docs/DECISIONS.md`.
- Use type hints and concise docstrings for public functions.
- Set deterministic seeds where practical.

## Required verification
After Python changes run:

```bash
make verify
```

## Definition of done
A task is complete only when:
1. the requested behavior is implemented;
2. relevant tests exist and pass;
3. the experiment command succeeds;
4. outputs contain enough metadata to reproduce the result;
5. limitations and assumptions are documented.

## Review checklist
- Is the reported compression ratio honest?
- Is there train/validation leakage?
- Are normalization statistics train-only?
- Are metrics computed after inverse transform where required?
- Are latitude weights applied correctly?
- Are all channel names and units preserved?
- Can the result be reproduced from one command?
