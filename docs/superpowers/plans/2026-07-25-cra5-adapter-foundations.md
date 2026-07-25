# CRA5 Adapter Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-safe data, evaluator, checkpoint-mapping, and runtime-bridge foundations required to adapt pretrained CRA5-159 to the canonical 28-channel ERA5 task.

**Architecture:** Keep scientific calculations and external-runtime concerns isolated. The main Python 3.12 package owns timestamp selection, normalization, evaluation, provenance, and bridge validation; an optional CRA5 process owns only checkpoint inference and entropy coding.

**Tech Stack:** Python 3.12, NumPy, pandas, PyTorch, xarray/Zarr, JSON/NPZ, pytest.

---

### Task 1: Leakage-Safe Nested Sample Manifests

**Files:**
- Modify: `src/era5_minimum/data/selection.py`
- Modify: `src/era5_minimum/data/seasonal_subsets.py`
- Create: `src/era5_minimum/data/sample_manifest.py`
- Modify: `tests/test_seasonal_subsets.py`
- Modify: `tests/test_time_splits.py`
- Create: `tests/test_sample_manifest.py`

- [ ] **Step 1: Write failing embargo and balance tests**

Add tests asserting:

```python
train = get_split_timestamps("train")
assert train.max() == pd.Timestamp("2019-12-24T18:00:00")
assert pd.Timestamp("2020-01-01") - train.max() > pd.Timedelta(days=6)

subsets = generate_nested_subsets([16, 32, 64, 128], seed=42)
assert subsets[16] == subsets[32][:16]
for n, values in subsets.items():
    frame = pd.DatetimeIndex(values)
    assert frame.month.map(month_to_season).value_counts().nunique() == 1
    assert frame.hour.value_counts().to_dict() == {0: n // 4, 6: n // 4, 12: n // 4, 18: n // 4}
    assert frame.year.value_counts().max() - frame.year.value_counts().min() <= 1
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_time_splits.py tests/test_seasonal_subsets.py`
Expected: failures because the current train pool reaches `2019-12-31` and does not balance UTC hour/year.

- [ ] **Step 3: Implement deterministic nested selection**

Set the training pool end to `2019-12-24T18:00:00`. Generate candidates in
balanced 16-item blocks over four seasons and four UTC hours, rotating six
training years deterministically. Reject non-positive sizes, sizes not
divisible by 16, duplicate sizes, and requests larger than the pool.

- [ ] **Step 4: Write and hash the manifest**

Implement:

```python
@dataclass(frozen=True)
class TrainingSampleManifest:
    seed: int
    temporal_embargo_hours: int
    source_uri: str
    subsets: dict[int, tuple[str, ...]]

    def to_dict(self) -> dict[str, object]: ...

def write_training_sample_manifest(
    path: Path,
    *,
    sizes: tuple[int, ...] = (16, 32, 64, 128),
    seed: int = 42,
) -> Path: ...
```

Write canonical sorted JSON containing split bounds, timestamps, nested-parent
size, SHA-256 over the payload without the hash field, and `train_only=true`.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest -q tests/test_time_splits.py tests/test_seasonal_subsets.py tests/test_sample_manifest.py`
Expected: all tests pass.

```bash
git add src/era5_minimum/data tests/test_time_splits.py tests/test_seasonal_subsets.py tests/test_sample_manifest.py
git commit -m "feat(data): add leakage-safe training sample manifests"
```

### Task 2: Scientific Local Evaluator

**Files:**
- Create: `src/era5_minimum/codec/evaluation.py`
- Create: `tests/test_codec_evaluation.py`
- Modify: `src/era5_minimum/codec/workflow.py`
- Modify: `tests/test_codec_smoke.py`
- Modify: `scripts/evaluate_codec_local.py`
- Create: `tests/test_evaluate_codec_local.py`

- [ ] **Step 1: Write failing metric-core tests**

Test a synthetic `[N,28,H,W]` tensor with known latitude-dependent errors and
validity masks. Require per-channel latitude-weighted physical RMSE, NRMSE by
train standard deviation, PSNR by train range, and equal-channel 8/20 grouped
scores.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_codec_evaluation.py`
Expected: import failure because `codec.evaluation` does not exist.

- [ ] **Step 3: Implement pure evaluation functions**

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

Return JSON-safe values. Perfect reconstruction and zero train range use
`psnr_db=null` with an explicit status instead of infinity.

- [ ] **Step 4: Add physical diagnostics**

Test and implement MSLP Pa/hPa, TP6H m/mm per 6h, each `Z*` and `Z/g` with
`g=9.80665`, U10, V10, and latitude-weighted wind-speed RMSE.

- [ ] **Step 5: Integrate one evaluator artifact**

Fit ranges from smoke train only. Write `local_evaluation.json` and derive
legacy `metrics_validation.json` and `metrics_per_channel.json` from it.
Include actual bytes, exact roundtrip, encode/decode seconds, git SHA, run ID,
and train-statistics checksum.

- [ ] **Step 6: Make the CLI strict**

The local evaluator CLI reads `local_evaluation.json`, validates
`evaluator_version`, `train_only=true`, and required metric groups, then prints
a compact summary. Missing provenance exits non-zero.

- [ ] **Step 7: Verify GREEN and commit**

Run:

```bash
pytest -q tests/test_codec_evaluation.py tests/test_codec_smoke.py tests/test_evaluate_codec_local.py
```

Expected: all tests pass.

```bash
git add src/era5_minimum/codec/evaluation.py src/era5_minimum/codec/workflow.py scripts/evaluate_codec_local.py tests/test_codec_evaluation.py tests/test_codec_smoke.py tests/test_evaluate_codec_local.py
git commit -m "feat(codec): add scientific local evaluator"
```

### Task 3: CRA5-159 Channel Mapping and Checkpoint Provenance

**Files:**
- Create: `src/era5_minimum/cra5/__init__.py`
- Create: `src/era5_minimum/cra5/channel_mapping.py`
- Create: `src/era5_minimum/cra5/provenance.py`
- Create: `scripts/fetch_cra5_checkpoint.py`
- Create: `tests/test_cra5_channel_mapping.py`
- Create: `tests/test_cra5_provenance.py`

- [ ] **Step 1: Write failing mapping tests**

Require CRA5-159 order:

```python
CRA5_PRESSURE_VARIABLES = ("z", "q", "u", "v", "t", "w")
CRA5_PRESSURE_LEVELS = (1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1)
CRA5_SURFACE_VARIABLES = ("v10", "u10", "v100", "u100", "t2m", "tcc", "sp", "tp6h", "msl")
```

Assert all 28 outputs remain in canonical order, exactly 26 have source
indices, and only `sst`/`tcwv` are marked `learned_boundary`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_cra5_channel_mapping.py`
Expected: import failure because the CRA5 package does not exist.

- [ ] **Step 3: Implement explicit mappings**

Create immutable `Cra5ChannelMapping` rows with:

```python
canonical_name: str
cra5_name: str | None
cra5_index: int | None
initialization: Literal["copy", "learned_boundary"]
```

Validate unique source indices and exact equality with `CHANNEL_NAMES`.

- [ ] **Step 4: Write failing provenance tests**

Use a temporary byte file to test streamed size/hash validation, cache-path
resolution, dry-run output, mismatch rejection, and refusal to accept a Git
LFS pointer as a checkpoint.

- [ ] **Step 5: Implement checkpoint manifest and fetch CLI**

Pin:

```python
CRA5_UPSTREAM_COMMIT = "2b9e06d8ca31f7b27c5039c9b5fc3334a43b4976"
CRA5_159_SIZE = 1_450_747_681
CRA5_159_SHA256 = "36dfdf0458bb9ed9ecfd1dcdbd75bf9d2599f2320041e769d6c766f3d8563a11"
```

The CLI defaults to dry-run. `--download` streams to a `.part` file under
`~/.cache/era5-minimum/cra5/`, verifies size/hash, then atomically renames it.
It never writes model bytes into the repository.

- [ ] **Step 6: Verify GREEN and commit**

Run: `pytest -q tests/test_cra5_channel_mapping.py tests/test_cra5_provenance.py`
Expected: all tests pass.

```bash
git add src/era5_minimum/cra5 scripts/fetch_cra5_checkpoint.py tests/test_cra5_channel_mapping.py tests/test_cra5_provenance.py
git commit -m "feat(cra5): add checkpoint provenance and channel mapping"
```

### Task 4: Versioned External Runtime Bridge

**Files:**
- Create: `src/era5_minimum/cra5/bridge.py`
- Create: `tests/fixtures/fake_cra5_runtime.py`
- Create: `tests/test_cra5_bridge.py`

- [ ] **Step 1: Write failing bridge tests**

Exercise a fake subprocess that reads a request JSON and writes a response JSON.
Require protocol version, operation, input/output paths, tensor shape, channel
order, checkpoint hash, bitstream SHA-256, exact roundtrip, and timings.
Reject path traversal, stale response run IDs, malformed JSON, timeout, and
non-zero process exit.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_cra5_bridge.py`
Expected: import failure because `cra5.bridge` does not exist.

- [ ] **Step 3: Implement bridge dataclasses and runner**

Implement `Cra5BridgeRequest`, `Cra5BridgeResponse`, and:

```python
def run_cra5_bridge(
    request: Cra5BridgeRequest,
    *,
    command: tuple[str, ...],
    work_dir: Path,
    timeout_seconds: float,
) -> Cra5BridgeResponse: ...
```

Use `subprocess.run(..., check=False, timeout=..., capture_output=True,
text=True)`. Resolve every artifact path under `work_dir` before reading it.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest -q tests/test_cra5_bridge.py`
Expected: all tests pass.

```bash
git add src/era5_minimum/cra5/bridge.py tests/fixtures/fake_cra5_runtime.py tests/test_cra5_bridge.py
git commit -m "feat(cra5): add isolated runtime bridge"
```

### Task 5: Foundation CLI and Documentation

**Files:**
- Create: `scripts/prepare_ml_sample_manifest.py`
- Modify: `Makefile`
- Modify: `docs/DECISIONS.md`
- Modify: `docs/DEVELOPMENT.md`
- Create: `tests/test_prepare_ml_sample_manifest.py`

- [ ] **Step 1: Write failing CLI test**

Run the script into a temporary directory and assert sizes 16/32/64/128,
manifest checksum, embargo, source URI, seed, and deterministic byte-identical
second output.

- [ ] **Step 2: Implement CLI and Make targets**

Add:

```text
make ml-samples
make cra5-checkpoint-dry-run
```

Neither target downloads the full ERA5 split or checkpoint by default.

- [ ] **Step 3: Document decisions**

Record DEC-030 for the scientific evaluator, DEC-031 for the seven-day train
embargo, and DEC-032 for isolated CRA5 transfer initialization and external
checkpoint provenance.

- [ ] **Step 4: Verify all foundations**

Run:

```bash
make verify
.venv/bin/python scripts/data/prepare_era5_28ch.py validate --dataset-dir data/era5_28ch_demo
.venv/bin/python scripts/fetch_cra5_checkpoint.py
.venv/bin/python scripts/prepare_ml_sample_manifest.py --output /tmp/era5-minimum-samples.json
```

Expected: repository verification passes, local demo validates, checkpoint
command reports dry-run provenance, and sample manifest is written.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_ml_sample_manifest.py Makefile docs/DECISIONS.md docs/DEVELOPMENT.md tests/test_prepare_ml_sample_manifest.py
git commit -m "feat(ml): expose CRA5 foundation workflow"
```
