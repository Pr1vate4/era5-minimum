# Grafana ML Observability Design

## Goal

Provide a one-command local monitoring stack in which the frontend Grafana
button opens a provisioned ERA5 model dashboard backed only by validated,
real model artifacts. The dashboard must reflect the hackathon requirements
without inventing absent VAEformer confidence intervals, spectral scores,
extreme-precipitation metrics, latent-probe results, GPU measurements, or
compression values.

## Current State

The repository already contains:

- `compose.monitoring.yaml` with Prometheus and Grafana;
- a provisioned Prometheus datasource;
- the `ERA5 API Overview` dashboard for HTTP and process telemetry;
- FastAPI Prometheus metrics and alert rules;
- a frontend Grafana button configured through `VITE_GRAFANA_URL`;
- confirmed N=32 model artifacts in `artifacts/model-n32/metrics.json` and
  `artifacts/model-n32/report.json`.

The missing link is a long-lived service that validates the model artifacts
and exposes a bounded Prometheus summary. Grafana currently has no model
dashboard and the frontend button opens the Grafana root instead of the model
view.

## Selected Architecture

Use a read-only JSON exporter rather than Pushgateway.

```text
artifacts/model-n32/*.json
          |
          v
ERA5 ML artifact exporter :9101/metrics
          |
          v
Prometheus :9090
          |
          v
Grafana /d/era5-model-overview/...
          ^
          |
frontend Grafana button
```

The exporter is a small Python module inside `src/era5_minimum/monitoring/`.
It uses the existing `prometheus_client` dependency and reads the artifact
directory configured by `ERA5_ML_METRICS_ROOT`. The Compose default is
`/workspace/artifacts/model-n32`, backed by the existing read-only
`${ARTIFACTS_DIR:-./artifacts}` mount.

Prometheus scrapes the exporter every five seconds. The exporter parses files
on scrape so replacing a valid artifact bundle updates Grafana without
rebuilding an image or restarting a container. A malformed or incomplete
bundle exports readiness/validation state only and logs the reason without
using exception text as a metric label.

## Alternatives Considered

### Pushgateway

Each training/evaluation command would push metrics when it exits. This works
for short-lived jobs but requires invasive changes to every CLI and careful
stale-series deletion. It is not selected for the hackathon handoff.

### API-only dashboard

The existing API dashboard could be expanded with request graphs, but it
cannot answer the scientific questions in the task specification. It remains
as the operational dashboard and is not treated as model evaluation.

## Artifact Validation

The exporter accepts a bundle only when all applicable invariants hold:

- `metrics.json` reports exactly the canonical 28-channel order;
- `metric_space` is `physical`;
- `latitude_weighting` and `train_only` are both true;
- the three NRMSE scores are finite;
- `report.json` identifies a real split and a positive frame count;
- `compression.total_bitstream_bytes` and
  `compression.compression_ratio` are positive;
- `exact_quantized_symbol_roundtrip_all_frames` is a boolean;
- tensor element ratio and serialized compression ratio remain separate;
- the selected target ratio is descriptive and is never substituted for the
  measured serialized ratio.

Optional fields are exported only when present and finite. Missing optional
fields produce Grafana `No data`; zero is not used as a placeholder. Artifact
paths, checkpoint paths, run IDs, exception messages, timestamps, and
filenames never become Prometheus labels.

The fixed 28 channel names are permitted labels for per-channel NRMSE and PSNR.
Their cardinality is bounded by the canonical channel contract.

## Prometheus Contract

Every accepted series uses bounded labels from `grid`, `target_cr`, and
`split` only where needed.

### Exporter state

- `era5_codec_artifact_ready`
- `era5_codec_artifact_last_modified_timestamp_seconds`
- `era5_codec_artifact_validation_failures_total`

### Compression and codec correctness

- `era5_codec_actual_compression_ratio`
- `era5_codec_tensor_compression_ratio`
- `era5_codec_bitstream_bytes`
- `era5_codec_bits_per_value`
- `era5_codec_exact_roundtrip`
- `era5_codec_frame_count`

### Scientific quality

- `era5_codec_overall_score`
- `era5_codec_surface_score`
- `era5_codec_pressure_score`
- `era5_codec_mean_psnr_db`
- `era5_codec_channel_nrmse{channel}`
- `era5_codec_channel_psnr_db{channel}`
- `era5_codec_mslp_rmse_hpa`
- `era5_codec_tp6h_rmse_mm_per_6h`
- `era5_codec_wind_speed_rmse_m_per_s`

### Runtime and data efficiency

- `era5_codec_encode_duration_seconds`
- `era5_codec_decode_duration_seconds`
- `era5_codec_encode_per_frame_seconds`
- `era5_codec_decode_per_frame_seconds`
- `era5_codec_unique_timestamps{split}`
- `era5_codec_optimizer_steps`
- `era5_codec_trainable_parameters`
- `era5_codec_training_duration_seconds`
- `era5_codec_peak_vram_bytes` when measured
- `era5_codec_gpu_hours` when measured

The exporter may scan multiple valid direct child directories under the
configured root later, but this implementation intentionally serves the one
selected bundle. This avoids presenting failed experiment-ladder entries as
scientific results.

## Grafana Dashboards

The existing `ERA5 API Overview` remains unchanged.

A new provisioned dashboard has:

- UID: `era5-model-overview`;
- title: `ERA5 Model — Compression & Quality`;
- default refresh: 10 seconds;
- default time range: six hours;
- Prometheus datasource UID: `prometheus`;
- direct local URL:
  `http://localhost:3000/d/era5-model-overview/era5-model-compression-quality`.

### Row 1: Result status

Six compact stat cards show artifact readiness, measured serialized
compression ratio, tensor ratio, exact roundtrip, total bitstream size, and
the number of evaluated frames. Serialized and tensor ratios have explicit,
different titles.

### Row 2: Scientific quality

A bar gauge compares overall, surface, and pressure NRMSE. Adjacent stats show
mean finite PSNR and the train/test timestamp counts. Threshold colors are
presentation aids for lower-is-better scores, not VAEformer pass/fail claims.

### Row 3: The 28-channel contract

Two horizontal bar-gauge panels display NRMSE and PSNR for all canonical
channels. This makes surface and pressure-level failures visible without
mixing physical units in one chart.

### Row 4: Physical diagnostics and speed

Stat panels show MSLP RMSE in hPa, TP6H RMSE in mm/6h, wind-speed RMSE in
m/s, encode/decode time per frame, and total encode/decode time.

### Row 5: Data and resources

Panels show unique train/validation timestamps, optimizer steps, trainable
parameters, training runtime, peak VRAM, and GPU hours. Missing GPU values
remain `No data`, which is correct for the current CPU artifact.

### Row 6: Required-but-unavailable evidence

A permanent text panel states that non-inferiority CI versus VAEformer,
spectral degradation, extreme-precipitation evaluation, and latent-probe
improvement are not present in the selected artifact. The dashboard must not
render pass/fail cards for these criteria until real evaluator outputs are
added to the validated exporter contract.

## Frontend Link and Local Access

`VITE_GRAFANA_URL` defaults to the direct model-dashboard URL. The existing
top-header button keeps opening a new tab.

For a frictionless local demo, Grafana anonymous access is enabled with Viewer
permissions only, sign-up remains disabled, and the host port binds to
`127.0.0.1` by default. Prometheus is also bound to loopback by default.
Admin credentials stay configurable for local dashboard inspection and are
never printed or committed.

## Compose and Commands

`compose.monitoring.yaml` adds `ml-metrics-exporter`, mounts artifacts
read-only, and makes Prometheus depend on its health check. The Prometheus
configuration gains a second scrape job named `era5-codec-artifacts`.

The supported commands remain:

```bash
make monitoring-up
make monitoring-ps
make monitoring-smoke
make monitoring-down
```

`make monitoring-up` starts frontend, API, exporter, Prometheus, and Grafana
through the existing merged Compose files. `monitoring-smoke` additionally
checks exporter readiness, the Prometheus target/query, the provisioned model
dashboard, and the anonymous direct dashboard URL.

## Error Handling

- Missing artifact root: exporter is healthy as a process but reports
  `era5_codec_artifact_ready 0`; Grafana shows `No data`.
- Invalid contract: readiness is zero, scientific series are omitted, and a
  bounded validation-failure counter increments once for each distinct invalid
  artifact modification, not once per Prometheus scrape.
- Prometheus unavailable: Grafana panels show datasource errors; the frontend
  continues to work.
- Grafana unavailable: the frontend link still opens the configured URL and
  does not claim Grafana is healthy.
- A previously valid bundle replaced by an invalid one is not served as
  current truth.

## Verification

Python tests cover:

- parsing the N=32-shaped artifact contract;
- rejection of noncanonical channel order;
- rejection of non-physical, non-latitude-weighted, or non-train-only metrics;
- rejection of missing serialized bytes/ratio or invalid roundtrip type;
- omission of optional metrics instead of zero filling;
- bounded per-channel series.

Monitoring tests cover:

- Prometheus configuration syntax;
- dashboard JSON syntax, UID, datasource, panel titles, and PromQL metric names;
- Compose rendering and read-only artifact mount;
- exporter `/health` and `/metrics`;
- Prometheus target and query;
- Grafana health and provisioned dashboard lookup;
- direct anonymous dashboard access.

After Python changes, the required repository command is `make verify`.
Container-level checks run through `make monitoring-check-prometheus-config`,
`make monitoring-check-rules`, and `make monitoring-smoke`.

## Scope Limits

- No model training or synthetic checkpoint generation.
- No changes to scientific metric definitions.
- No claim of VAEformer non-inferiority without reference artifacts and block
  bootstrap output.
- No Alertmanager or external notification delivery.
- No public Grafana deployment; this is a loopback-only local stack.
- No datasets, checkpoints, bitstreams, generated dashboards, or monitoring
  volumes are committed.
