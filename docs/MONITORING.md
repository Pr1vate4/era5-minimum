# Monitoring

The local monitoring path is FastAPI `/metrics` → Prometheus → Grafana. It is
available without ERA5 files, a neural codec, a GPU, or model checkpoints.

## Start and verify

```bash
cp .env.example .env
# Change GRAFANA_ADMIN_PASSWORD in .env outside local development.
make monitoring-up
make monitoring-ps
make monitoring-smoke
```

Default local addresses are:

| Service | Address |
| --- | --- |
| API | `http://localhost:${API_PORT}` |
| Metrics | `http://localhost:${API_PORT}/metrics` |
| Prometheus | `http://localhost:${PROMETHEUS_PORT}` |
| Grafana | `http://localhost:${GRAFANA_PORT}` |

Grafana uses `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` from `.env`
(the example defaults are `admin` / `change-me`). Never commit `.env` or a real
password. The Prometheus datasource (UID `prometheus`) and the **ERA5 API
Overview** dashboard (UID `era5-api-overview`) are provisioned at startup; no
manual UI import is required.

Run `make monitoring-check-prometheus-config` and
`make monitoring-check-rules` to use `promtool` from the configured official
Prometheus image. The stack uses named `prometheus_data` and `grafana_data`
volumes. `make monitoring-down` and `make monitoring-clean` intentionally keep
them. Remove those named volumes explicitly only when monitoring history and
Grafana state may be discarded; do not use broad Docker prune commands.

## API metrics

| Metric | Labels | Meaning |
| --- | --- | --- |
| `era5_api_http_requests_total` | `method`, `route`, `status_code` | Completed API request. |
| `era5_api_http_request_duration_seconds` | `method`, `route` | Request duration histogram. |
| `era5_api_http_requests_in_progress` | `method` | Current requests in flight. |
| `era5_api_artifact_load_total` | `artifact_type`, `status` | Successful or failed artifact read/load. |
| `era5_api_artifact_validation_total` | `artifact_type`, `status` | Valid or invalid artifact validation. |
| `era5_api_artifact_loaded` | `artifact_type` | Last artifact-load state (0 or 1). |
| `era5_api_artifacts_loaded` | — | Current count of successfully loaded artifact types. |
| `era5_api_artifact_last_success_timestamp_seconds` | `artifact_type` | Last successful artifact load time. |
| `era5_api_build_info` | `version`, `commit`, `environment` | Static low-cardinality build metadata. |

Artifact types are strictly `summary`, `experiments`, `sample_efficiency`, and
`reconstruction`. Artifact reads and Pydantic validation in the real repository
layer produce the metrics; no mock endpoint is used. A successful validated
load increments both `load_total{status="success"}` and
`validation_total{status="valid"}`. Read failures set `artifact_loaded` to 0;
validation failures also set it to 0.

The request-duration buckets are 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1,
2.5, 5, 10 and 30 seconds. `/metrics` is excluded from the HTTP metrics.
The optional response-size histogram is deliberately absent: calculating it
correctly would require intercepting bodies and must not buffer or break a
future streaming response.

Prometheus also exports supported Python process metrics such as
`process_cpu_seconds_total`, `process_resident_memory_bytes`,
`process_open_fds`, `process_max_fds`, and Python garbage-collection metrics.

## PromQL examples

```promql
sum(rate(era5_api_http_requests_total[1m]))

100 * sum(rate(era5_api_http_requests_total{status_code=~"5.."}[5m]))
  / clamp_min(sum(rate(era5_api_http_requests_total[5m])), 1e-9)

histogram_quantile(0.95, sum by (le) (
  rate(era5_api_http_request_duration_seconds_bucket[5m])
))
```

The bundled rules expose `Era5ApiDown`, `Era5ApiHighErrorRate`, and
`Era5ApiHighP95Latency` in Prometheus. There is no Alertmanager in this task,
so these rules do not send notifications yet.

## Cardinality and runtime limits

`route` always uses a FastAPI route template (for example,
`/api/v1/experiments/{experiment_id}`), while an unknown path is the bounded
value `unmatched`. Query strings, filenames, artifact IDs, timestamps, paths,
request IDs, exception text, IP addresses, and user agents are never labels.
They would create unbounded time series and can make Prometheus unavailable.

The Compose API command intentionally uses one Uvicorn worker. The default
Prometheus registry is suitable for that single process. Do not increase it
without a separately implemented and verified `prometheus_client`
multiprocess setup, including `PROMETHEUS_MULTIPROC_DIR` lifecycle handling.

`/metrics` is unauthenticated for local Compose scraping. Do not expose it
directly to the public internet; place it behind suitable network controls and
authentication in a deployed environment.

## ML metrics are deferred

No neural codec metric, training result, compression ratio, or GPU statistic is
fabricated by this stack. Prometheus is not a replacement for
`resource_usage.json`, `run_summary.json`, or detailed results JSON. The future
contract and two supported integration approaches (Pushgateway or a JSON
exporter) are documented in [ML_METRICS_CONTRACT.md](ML_METRICS_CONTRACT.md).
Pushgateway is intentionally not a service in the current Compose stack.
