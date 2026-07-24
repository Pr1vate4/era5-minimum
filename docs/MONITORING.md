# Backend monitoring

The Artifact API exports Prometheus metrics at `/metrics` through the official
`prometheus_client.make_asgi_app` ASGI adapter. The endpoint uses the standard
Prometheus exposition format and its own scrapes are excluded from the API HTTP
metrics.

## Start locally

Install the API server extra, then run exactly one Uvicorn worker:

```bash
python -m pip install -e ".[api]"
python -m uvicorn era5_minimum.api.app:app --host 127.0.0.1 --port 8000 --workers 1
curl -fsS http://127.0.0.1:8000/metrics
```

There is no committed Gunicorn or multi-worker launch configuration. The
default Prometheus registry is correct for this single-process local command.
Do not run multiple workers for a shared scrape target until official
`prometheus_client` multiprocess mode, including `PROMETHEUS_MULTIPROC_DIR`
lifecycle management, is configured separately.

## API metrics

| Metric | Labels | Purpose |
| --- | --- | --- |
| `era5_api_http_requests_total` | `method`, `route`, `status_code` | Completed user HTTP requests. |
| `era5_api_http_request_duration_seconds` | `method`, `route` | User request duration histogram. |
| `era5_api_http_requests_in_progress` | `method`, `route` | Requests currently in flight. |

`route` is the FastAPI route template, for example
`/api/v1/experiments/{experiment_id}`. Unknown paths use `unmatched`.
Actual URL values, query strings, timestamps, filenames, exception text and
tracebacks are never labels.

The duration histogram uses local API buckets: 0.005, 0.01, 0.025, 0.05, 0.1,
0.25, 0.5, 1, 2.5, 5 and 10 seconds.

## Artifact metrics

| Metric | Labels | Purpose |
| --- | --- | --- |
| `era5_api_artifact_load_errors_total` | `artifact_type` | JSON file absent or malformed during repository loading. |
| `era5_api_artifact_validation_errors_total` | `artifact_type` | Artifact fails repository/Pydantic validation. |
| `era5_api_artifacts_loaded_total` | `artifact_type` | JSON artifact was read and validated successfully. |

`artifact_type` is bounded to `summary`, `experiments`, `sample_efficiency`,
and `reconstruction`; it never contains an artifact filename or experiment ID.

## Standard process metrics

The default `prometheus-client` registry exposes process metrics on supported
platforms, including `process_cpu_seconds_total`,
`process_resident_memory_bytes`, `process_open_fds`, and
`process_start_time_seconds`. The API does not reimplement them.
