# ERA5 model/codec acceptance

`scripts/model/accept_codec.py` is a model-level, checkpoint-backed acceptance
command. It does not train, download ERA5, alter a Zarr store, or call the API.
It processes one timestamp at a time.

## Preconditions

The real demo is `data/era5_28ch_demo`, validation timestamps on 2020-01-01 at
00:00, 06:00, 12:00 and 18:00 UTC. It is evaluation data, not a source of
normalization statistics.

A compatible checkpoint is required. It must contain these fields:

- `state_dict`, loading with `strict=True` into `ConvAutoencoder`;
- `model_config.in_channels=28` and `model_config.latent_channels`;
- `codec_config.version` and `codec_config.quantization_step`;
- `channel_order` in the exact official order;
- `normalization` containing 28 train-only mean/std values and a source
  manifest SHA256.

The strict channel order is:

`t2m, mslp, u10, v10, tp6h, sst, tcwv, tcc, T1000, T925, T850, T700, U1000,
U925, U850, U700, V1000, V925, V850, V700, Z1000, Z925, Z850, Z700, Q1000,
Q925, Q850, Q700`.

No checkpoint in the repository means acceptance is `BLOCKED`; this command
does not substitute random weights, a synthetic checkpoint, or validation
statistics.

## What is measured

For every input, the command performs:

`physical fields → SST ocean mask → train-only normalization → encoder →`
`uniform scalar quantization → canonical Huffman byte stream → new codec`
`context → entropy decode → decoder → denormalization → metrics`.

The physical Zarr retains NaN SST values over land. Invalid values are filled
only at the model boundary; after decode, SST land cells are restored to NaN.

Every standalone `.e5ac` stream includes its magic, JSON header, latent and
original shapes, checkpoint checksum, canonical channel order, quantization
metadata, and entropy payload. A decoder needs the recorded checkpoint, but
checkpoint bytes are not charged per sample.

The only reported binary compression ratio is:

`(32 × T × C × H × W) / (8 × total_bitstream_bytes)`.

`total_bitstream_bytes` includes header and all per-sample side information.
Latent element reduction is deliberately not called a compression ratio.

## Commands

Check the planned real-dataset invocation without loading a model:

```bash
source .venv/bin/activate
python scripts/model/accept_codec.py \
  --dataset-root data/era5_28ch_demo --split validation \
  --checkpoint checkpoints/era5_codec.ckpt \
  --output-dir outputs/model_acceptance/demo --timestamps all --dry-run
```

Run a 128×128 CPU crop smoke when a compatible checkpoint is available:

```bash
python scripts/model/accept_codec.py \
  --dataset-root data/era5_28ch_demo --split validation \
  --checkpoint checkpoints/era5_codec.ckpt \
  --output-dir outputs/model_acceptance/crop-128 --timestamp 2020-01-01T00:00:00.000000000 \
  --crop-height 128 --crop-width 128 --device cpu --deterministic
```

Run four sequential full frames only after a crop memory check:

```bash
python scripts/model/accept_codec.py \
  --dataset-root data/era5_28ch_demo --split validation \
  --checkpoint checkpoints/era5_codec.ckpt \
  --output-dir outputs/model_acceptance/full-demo --timestamps all \
  --full-frame --max-samples 4 --device cuda --deterministic
```

The command refuses a non-empty output by default. `--overwrite` only permits
replacement inside the specified `outputs/model_acceptance/...` directory; it
never targets datasets. `--skip-bitstream` is a structural diagnostic and
cannot produce an accepted codec result.

## Artifacts and interpretation

The output has `acceptance_report.md`, model and dataset provenance, run
configuration, per-timestamp metrics, timing/resource records, compression
accounting and exact roundtrip checks. `bitstreams/` contains only generated
per-sample streams; no full original/reconstruction arrays are persisted.

Metrics are calculated in physical units per channel using valid-point,
latitude weights proportional to `cos(latitude)`. SST is ocean-only. NRMSE is
reported only from the checkpoint's declared train-only standard deviation.
The cross-variable surface/pressure/overall scores remain unavailable until a
dimensionless aggregation convention is predeclared; they are not invented by
the acceptance command.

`crop_smoke` is explicitly not global verification. A `PASS` requires full
grid, a valid checkpoint, real serialized bytes, exact quantized-symbol
roundtrip, independent decode and physical metrics. A successful crop is
`PASS_WITH_LIMITATIONS`; missing checkpoint/config/train-only normalization is
`BLOCKED`.

## Verification and next steps

Unit tests use a tiny deterministic checkpoint fixture only to exercise the
interface; it is not a trained ERA5 model. Run them without GPU:

```bash
python -m pytest -q tests/test_codec_acceptance.py tests/test_codec_harness.py
python -m pip check
```

The current real demo cannot support a final quality claim or the planned
N=128 experiment: that needs a compatible trained checkpoint, recorded
train-only statistics, immutable train/validation/test timestamp splits and a
separate training decision. Backend `mode=reconstructed` and `mode=error`
should be connected later only after a `PASS` artifact defines a stable model
and bitstream/checkpoint provenance contract.
