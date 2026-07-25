# Future ML metrics contract

This document specifies a future Prometheus interface only. ERA5-Minimum does
not register these metrics until real codec CLIs and their artifact formats
exist. Patch PCA is not a neural codec; its tensor-element comparison must not
be published as an actual serialized compression ratio.

## Proposed metrics

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `era5_codec_runs_total` | Counter | `operation`, `status`, `grid`, `target_cr` | Completed CLI attempts. |
| `era5_codec_last_run_success` | Gauge | `operation`, `grid`, `target_cr` | Most recent run succeeded (1/0). |
| `era5_codec_last_run_timestamp_seconds` | Gauge | `operation`, `grid`, `target_cr` | Most recent run completion time. |
| `era5_codec_run_duration_seconds` | Histogram | `operation`, `grid`, `target_cr`, `status` | Duration of a real completed CLI run. |
| `era5_codec_optimizer_steps` | Gauge | `grid`, `target_cr` | Last actual optimizer step count. |
| `era5_codec_unique_timestamps` | Gauge | `grid`, `target_cr`, `split` | Distinct timestamps in the real split. |
| `era5_codec_peak_vram_bytes` | Gauge | `operation`, `grid`, `target_cr` | Measured peak GPU memory. |
| `era5_codec_actual_compression_ratio` | Gauge | `grid`, `target_cr`, `split` | Real serialized compression ratio only. |
| `era5_codec_bitstream_bytes` | Gauge | `grid`, `target_cr`, `split` | Actual bitstream size. |
| `era5_codec_encode_duration_seconds` | Gauge | `grid`, `target_cr`, `split` | Latest measured encode duration. |
| `era5_codec_decode_duration_seconds` | Gauge | `grid`, `target_cr`, `split` | Latest measured decode duration. |
| `era5_codec_exact_roundtrip` | Gauge | `grid`, `target_cr`, `split` | Exact quantized-symbol roundtrip (1/0). |
| `era5_codec_overall_score` | Gauge | `grid`, `target_cr`, `split` | Confirmed overall evaluation score. |
| `era5_codec_surface_score` | Gauge | `grid`, `target_cr`, `split` | Confirmed surface-field score. |
| `era5_codec_pressure_score` | Gauge | `grid`, `target_cr`, `split` | Confirmed pressure-field score. |

Allowed `operation` values are `train`, `encode`, `decode`, `evaluate`, and
`probe`. Allowed `status` values are `success`, `error`, `cancelled`, and
`limit_exceeded`. `target_cr` is a declared experimental target, not evidence
of achieved compression. The histogram records an observation only after a
real run ends; it is not a gauge of a made-up zero-duration run.

Never use `run_id`, checkpoint or bitstream paths, filenames, timestamps,
channels, or exception messages as Prometheus labels. Per-channel metrics are
also deferred until their cardinality is evaluated.

Detailed, reproducible evidence remains in `resource_usage.json`,
`run_summary.json`, `metrics_validation.json`, `metrics_per_channel.json`,
`metrics_per_time.json`, and `bitstream_statistics.json`. Prometheus is a
small operational summary, not an experiment archive.

## Future CLI integration

For short-lived `train`, `encode`, `decode`, and `evaluate` commands, a future
Pushgateway integration may push the last confirmed result. It must use a
stable grouping key, remove stale metrics, never use `run_id` as a label, and
must not replace the JSON artifacts above.

Alternatively, a long-lived JSON exporter may read confirmed
`resource_usage.json` and `run_summary.json` files and export only their latest
valid summary. Neither integration is implemented and no Pushgateway service
is added until real CLI outputs and schemas are available.
