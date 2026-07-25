# Local Codec Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Расширить предварительный локальный evaluator ML-001 до latitude-weighted per-channel метрик, train-only PSNR и обязательных физических diagnostics из раздела 14 `TASKA_ML.txt`.

**Architecture:** Чистые вычисления вынести в `codec/evaluation.py`, чтобы workflow и CLI использовали один контракт. Workflow вычисляет диапазоны только на train split, сохраняет полный evaluation artifact и совместимые legacy JSON; CLI только валидирует и суммирует сохраненный отчет, не переобучая статистики на validation/test.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest, JSON CLI.

---

### Task 1: Scientific metric core

**Files:**
- Create: `src/era5_minimum/codec/evaluation.py`
- Create: `tests/test_codec_evaluation.py`

- [ ] **Step 1: Write failing latitude-weighting and grouping tests**

Add tests that construct `[N,C,H,W]` physical tensors with larger polar error, masks, 8 surface and 20 pressure channels. Assert that each channel uses cosine-latitude weighted RMSE, NRMSE divides by train std, and `overall_score == 0.5 * surface_score + 0.5 * pressure_score`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_codec_evaluation.py`
Expected: collection fails because `era5_minimum.codec.evaluation` does not exist.

- [ ] **Step 3: Implement train-only ranges and per-channel evaluation**

Implement:

```python
def fit_train_channel_ranges(
    train_physical: np.ndarray,
    *,
    validity_mask: np.ndarray,
) -> np.ndarray: ...

def evaluate_reconstruction(
    original_physical: np.ndarray,
    reconstruction_physical: np.ndarray,
    *,
    latitudes: np.ndarray,
    channel_order: tuple[str, ...],
    train_std: np.ndarray,
    train_ranges: np.ndarray,
    validity_mask: np.ndarray,
) -> dict[str, Any]: ...
```

Validate shapes and positive train std. Compute per-channel latitude-weighted physical RMSE, NRMSE, PSNR from train-only range, equal-channel group scores, and explicit `None` PSNR status for zero RMSE or zero range so JSON never emits infinity.

- [ ] **Step 4: Write failing physical diagnostics tests**

Add exact conversion tests for `mslp` Pa/hPa, `tp6h` m/mm per 6h, every `Z*` channel and `Z/g` using `g=9.80665`, `u10`, `v10`, and wind-speed RMSE from vector magnitudes.

- [ ] **Step 5: Implement diagnostics and verify GREEN**

Run: `pytest -q tests/test_codec_evaluation.py`
Expected: all evaluator tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/era5_minimum/codec/evaluation.py tests/test_codec_evaluation.py
git commit -m "feat(codec): add scientific local evaluator"
```

### Task 2: Workflow artifacts

**Files:**
- Modify: `src/era5_minimum/codec/workflow.py`
- Modify: `tests/test_codec_smoke.py`

- [ ] **Step 1: Write failing artifact contract test**

Extend smoke tests to require `local_evaluation.json`; latitude-weighted per-channel fields; train-only range provenance; PSNR statuses; physical diagnostics; exact roundtrip; actual compression ratio; and encode/decode seconds.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_codec_smoke.py -k local_evaluation`
Expected: failure because the artifact is absent.

- [ ] **Step 3: Integrate the evaluator**

Fit channel ranges from `train_raw` with the train validity mask, call the pure evaluator for validation reconstruction, merge codec/runtime fields, save `local_evaluation.json`, and derive `metrics_validation.json` plus `metrics_per_channel.json` from the same result.

- [ ] **Step 4: Persist provenance**

Save train ranges and a SHA-256 provenance checksum in checkpoint/checkpoint metadata. Keep normalization train-only and SST ocean masking unchanged.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_codec_smoke.py`
Expected: all smoke tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/era5_minimum/codec/workflow.py tests/test_codec_smoke.py
git commit -m "feat(codec): persist local evaluation artifacts"
```

### Task 3: CLI and research decision

**Files:**
- Modify: `scripts/evaluate_codec_local.py`
- Create: `tests/test_evaluate_codec_local.py`
- Modify: `docs/DECISIONS.md`

- [ ] **Step 1: Write failing CLI tests**

Run the CLI against a minimal temporary run directory. Assert that it reads `local_evaluation.json`, prints the score/PSNR/physical diagnostic summary, and fails clearly if required provenance or evaluator status is missing.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evaluate_codec_local.py`
Expected: failure because the CLI still reads only legacy summary files.

- [ ] **Step 3: Implement strict report loading**

Load and validate `local_evaluation.json`; expose `run_id`, actual ratio, exact roundtrip, timings, grouped scores, mean finite PSNR, physical diagnostics, and the explicit preliminary-evaluator limitation.

- [ ] **Step 4: Document DEC-030**

Record that ranges/std are train-only, every spatial RMSE is latitude weighted, groups are equal-channel 8/20 means, and this module does not replace EVAL-001 spectral/bootstrap/extreme-precipitation evaluation.

- [ ] **Step 5: Verify the block and repository**

Run:

```bash
pytest -q tests/test_codec_evaluation.py tests/test_codec_smoke.py tests/test_evaluate_codec_local.py
make verify
```

Expected: all tests and repository verification pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_codec_local.py tests/test_evaluate_codec_local.py docs/DECISIONS.md
git commit -m "feat(codec): expose local evaluator report"
```
