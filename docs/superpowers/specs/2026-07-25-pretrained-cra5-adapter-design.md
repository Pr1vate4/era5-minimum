# Pretrained CRA5 Adapter Design

**Status:** approved
**Date:** 2026-07-25
**Owner:** ML-001

## Goal

Build a reproducible 28-channel ERA5 neural codec candidate by transferring the
public CRA5/VAEformer checkpoint instead of training a large weather
autoencoder from scratch. Measure the minimum additional ERA5 sample count
needed to reach useful 32x and 64x operating points.

The design must preserve the repository rules: no temporal leakage, train-only
normalization, latitude-weighted physical metrics, exact entropy roundtrip, and
separate tensor and serialized compression ratios.

## Why this path

The official CRA5 project publishes VAEformer source, entropy coding APIs, and
pretrained 159-variable and 268-variable checkpoints. The 159-variable
training config uses ERA5 through 2017, while this project fixes validation to
2020 and test to 2021. This makes the 159-variable checkpoint the preferred
transfer source, subject to recording that the temporal coverage is inferred
from the upstream config rather than asserted by the model card.

Sources:

- <https://github.com/taohan10200/CRA5>
- <https://arxiv.org/abs/2405.03376>
- <https://huggingface.co/taohan10200/CRA5-model>

The checkpoint is an initialization, not a result. Competition metrics are
computed only after adapting it to this repository's exact 28-channel contract
and fixed splits.

## Architecture

The implementation has four bounded components.

### 1. Dataset sampler

The existing WeatherBench2 reader remains the source adapter. A new sampler
creates deterministic timestamp manifests for sample counts `16`, `32`, `64`,
and `128` from 2014 through 2019-12-24. The final seven days of 2019 form a
temporal embargo before validation starts. Selection is balanced across:

- four meteorological seasons;
- four UTC synoptic times;
- all six training years.

The sampler stores timestamps and a SHA-256 digest before any field values are
read. It rejects any selected training timestamp less than seven days before a
validation timestamp. Validation remains 2020 and test remains 2021. The
four-frame local demo is only a technical validation smoke set and is never
used for fitting.

The first implementation downloads only selected frames. It must not launch a
multi-terabyte full-split download.

### 2. CRA5 checkpoint probe and adapter

The preferred source is `cra5_159v_150k.pth`:

```text
sha256:36dfdf0458bb9ed9ecfd1dcdbd75bf9d2599f2320041e769d6c766f3d8563a11
size:1450747681
```

The file is cached outside Git under
`~/.cache/era5-minimum/cra5/`. It is never copied into a run artifact or
repository. A manifest records URL, expected size, expected hash, upstream
commit, and license note.

CRA5-159 uses six pressure variables at 25 levels plus nine surface variables.
Twenty-six of the required channels can be mapped directly:

- pressure: `Z`, `Q`, `U`, `V`, `T` at 1000, 925, 850, and 700 hPa;
- surface: `t2m`, `mslp`, `u10`, `v10`, `tp6h`, and `tcc`.

`sst` and `tcwv` have no direct CRA5-159 source channel. They must never be
silently substituted. Their boundary weights start deterministically from
zero and are learned only from the selected training split.

The adapted model keeps VAEformer latent and hyperprior blocks. Its input patch
embedding and output projection are changed from 159 to 28 channels:

- mapped input kernels and output rows are copied by explicit channel index;
- `sst` and `tcwv` boundary parameters use deterministic zero initialization;
- all copied interior parameters are frozen for the first stage;
- the input embedding, output projection, and train-only affine adapter are
  optimized first;
- entropy/hyperprior parameters may be fine-tuned only after reconstruction is
  stable.

No upstream CRA5 normalization file is used for evaluation or fitting.
Normalization statistics are recomputed from each selected training subset.

### 3. Isolated CRA5 runtime

CRA5 upstream pins an older PyTorch stack and builds C++ entropy extensions.
It is not added to the default project environment until compatibility is
proven. Integration uses a versioned bridge:

- the main Python 3.12 process writes a strict request manifest and tensors;
- an explicitly configured CRA5 runtime performs model operations;
- the bridge returns bitstreams, reconstructions, timings, and provenance;
- both sides validate shapes, channel order, hashes, and protocol version.

Unit tests use a deterministic fake bridge. A result is not marked
`cra5_runtime_verified` until a real checkpoint encode/decode command succeeds
in the GPU/runtime environment.

The long-term preferred deployment is a pinned GPU container or isolated venv.
The bridge prevents old CRA5 dependencies from changing this project's
canonical `pyproject.toml` environment.

### 4. Evaluator and experiment ladder

One evaluator handles PCA, the current ConvAE, and the CRA5 adapter. For every
validation/test reconstruction it saves:

- latitude-weighted RMSE in physical units for every channel;
- NRMSE using train-only standard deviation;
- train-only-range PSNR;
- equal-channel surface score over 8 channels;
- equal-channel pressure score over 20 channels;
- `overall = 0.5 * surface + 0.5 * pressure`;
- MSLP Pa/hPa, TP6H m/mm, Z and Z/g, U10, V10, and wind-speed diagnostics;
- encode/decode time;
- exact quantized-symbol roundtrip;
- actual serialized bitstream bytes and compression ratio;
- tensor latent reduction as a separate descriptive metric.

The experiment ladder is:

1. Patch PCA at 32x and 64x as a near-training-free reference.
2. Existing ConvAE smoke as a pipeline reference, never as a real-data claim.
3. CRA5 adapter with `n=16`.
4. CRA5 adapter with `n=32`, `64`, and `128` only while quality improves.
5. Optional interior unfreezing only if boundary-only adaptation misses the
   validation target.

Checkpoint selection uses validation only. Test is read exactly once for the
selected configuration. The minimum sample count is the smallest count meeting
the target whose next larger count does not produce a material validation
improvement under the configured stopping rule.

## Bitstream contract

The serialized ratio is:

```text
original float32 bytes / total decoder-required serialized bytes
```

The denominator includes latent strings, hyperprior strings, headers, shape
metadata, and any per-run decoder-required side information. Shared fixed model
weights are reported separately and are not added per timestamp, matching the
deployed-codec interpretation. A second amortized ratio may include checkpoint
bytes, but it must have an explicit name and dataset-size denominator.

Exact roundtrip means the decoder receives exactly the quantized latent symbols
produced by the encoder. It does not mean lossy physical reconstruction equals
the original tensor.

## Failure policy

- Missing or mismatched checkpoint hash: stop before model construction.
- Unknown CRA5 state-dict layout: stop with the first incompatible key/shape.
- Missing channel mapping: stop; never reorder or substitute channels.
- Validation/test timestamp in a fitting manifest: stop.
- Upstream temporal coverage cannot be justified: mark transfer results
  ineligible for leakage-safe comparison until resolved.
- CRA5 runtime unavailable: continue PCA/evaluator/data work and emit a clear
  blocked runtime report, not fake neural metrics.
- Target serialized ratio not reached: report the measured ratio and do not
  label the run 32x or 64x.

## Testing

Required automated coverage:

- canonical CRA5-159 to ERA5-28 channel indices;
- explicit unsupported `sst` and `tcwv`;
- deterministic seasonal/synoptic sample manifests;
- split and adjacency leakage rejection;
- checkpoint hash/size/provenance validation;
- bridge request/response schema and failure handling;
- mapped checkpoint tensor slicing on a synthetic state dict;
- train-only normalization and SST ocean mask;
- evaluator formulas and physical diagnostics;
- bitstream byte accounting and exact symbol roundtrip;
- reproducible artifact metadata.

A real runtime smoke must encode and decode one local validation timestamp before
any training experiment is launched.

## Deliverables

- deterministic remote-subset preparation command;
- checkpoint provenance/fetch command;
- CRA5 bridge protocol and adapter;
- 28-channel local evaluator;
- configs for 16/32/64/128 sample experiments and 32x/64x targets;
- comparison artifacts for Patch PCA, ConvAE, and CRA5 adapter;
- one orchestration command that records config, seed, git SHA, runtime,
  checkpoint provenance, metrics, and limitations.

## Explicit limitations

- The current local archive has four validation timestamps and no training
  split. It validates I/O only.
- Resource limits are not yet confirmed by organizers, so real VAEformer GPU
  execution remains a gated smoke test.
- Spectral metrics, bootstrap confidence intervals, and extreme-precipitation
  evaluation remain official-evaluator work unless implemented separately.
- External checkpoint provenance must be reported; pretrained performance is
  never presented as training from the selected local sample count.
