# Grafana ML Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start Grafana with the local stack, expose validated real model artifacts to Prometheus, open the model dashboard from the frontend, and provision hackathon-relevant graphs without fabricated metrics.

**Architecture:** A read-only Python exporter validates `metrics.json` and `report.json` from the selected model artifact directory and exposes a bounded Prometheus contract. Prometheus scrapes API and model exporters; Grafana provisions separate runtime and model dashboards. The frontend opens the model dashboard directly.

**Tech Stack:** Python 3.12, `prometheus_client`, pytest, Docker Compose, Prometheus, Grafana 12, React/Vite, TypeScript.

---

### Task 1: Validate model artifacts and map real metric samples

**Files:**
- Create: `src/era5_minimum/monitoring/__init__.py`
- Create: `src/era5_minimum/monitoring/ml_artifacts.py`
- Create: `tests/test_ml_metrics_exporter.py`

- [ ] **Step 1: Write the failing parser tests**

Create a minimal fixture containing the canonical `CHANNEL_NAMES`, physical,
latitude-weighted, train-only evaluation metadata, group scores, per-channel
NRMSE/PSNR, physical diagnostics, training metadata, measured compression,
timings, and exact roundtrip.

```python
def test_load_snapshot_keeps_serialized_and_tensor_ratios_separate(tmp_path):
    root = write_valid_bundle(tmp_path)
    snapshot = load_ml_artifact_snapshot(root)
    assert snapshot.values["actual_compression_ratio"] == 135.6
    assert snapshot.values["tensor_compression_ratio"] == 32.0
    assert snapshot.values["bitstream_bytes"] == 3423714
    assert snapshot.values["exact_roundtrip"] == 1.0


def test_load_snapshot_rejects_noncanonical_channels(tmp_path):
    root = write_valid_bundle(tmp_path, channel_order=list(reversed(CHANNEL_NAMES)))
    with pytest.raises(MlArtifactContractError, match="channel order"):
        load_ml_artifact_snapshot(root)


def test_optional_resource_metrics_are_omitted_not_zero_filled(tmp_path):
    snapshot = load_ml_artifact_snapshot(write_valid_bundle(tmp_path))
    assert "peak_vram_bytes" not in snapshot.values
    assert "gpu_hours" not in snapshot.values
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run:

```bash
python -m pytest tests/test_ml_metrics_exporter.py -q
```

Expected: import failure because `era5_minimum.monitoring.ml_artifacts` does
not exist.

- [ ] **Step 3: Implement the strict artifact parser**

Implement:

```python
@dataclass(frozen=True)
class MlArtifactSnapshot:
    labels: dict[str, str]
    values: dict[str, float]
    channel_nrmse: dict[str, float]
    channel_psnr_db: dict[str, float]
    last_modified_timestamp: float


class MlArtifactContractError(ValueError):
    pass


def load_ml_artifact_snapshot(root: Path) -> MlArtifactSnapshot:
    metrics = read_object(root / "metrics.json")
    report = read_object(root / "report.json")
    validate_channel_order(metrics["metrics"]["channel_order"], CHANNEL_NAMES)
    require(metrics["metrics"]["metric_space"] == "physical", "metric_space")
    require(metrics["metrics"]["latitude_weighting"] is True, "latitude weighting")
    require(metrics["metrics"]["train_only"] is True, "train_only")
    require(type(report["exact_quantized_symbol_roundtrip_all_frames"]) is bool, "roundtrip")
    # Read measured serialized fields from report.compression and tensor ratio
    # from report.tensor_element_ratio; never derive one from the other.
```

Validate finite positive serialized bytes/ratio, finite scores, positive frame
count, and unique channel entries. Map the current `artifacts/model-n32`
schema, including MSLP hPa, TP6H mm/6h, wind-speed RMSE, timestamp counts,
steps, parameters, training runtime, and optional GPU values.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_ml_metrics_exporter.py -q
```

Expected: all parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/era5_minimum/monitoring tests/test_ml_metrics_exporter.py
git commit -m "feat: validate ML monitoring artifacts"
```

### Task 2: Expose the validated snapshot through Prometheus

**Files:**
- Create: `src/era5_minimum/monitoring/ml_exporter.py`
- Modify: `tests/test_ml_metrics_exporter.py`

- [ ] **Step 1: Write failing exposition tests**

```python
def test_collector_exposes_real_codec_metrics(tmp_path):
    registry = CollectorRegistry()
    registry.register(MlArtifactCollector(write_valid_bundle(tmp_path)))
    text = generate_latest(registry).decode()
    assert "era5_codec_actual_compression_ratio" in text
    assert "era5_codec_tensor_compression_ratio" in text
    assert 'era5_codec_channel_nrmse{channel="t2m"}' in text


def test_invalid_bundle_exposes_readiness_only(tmp_path):
    registry = CollectorRegistry()
    registry.register(MlArtifactCollector(tmp_path / "missing"))
    text = generate_latest(registry).decode()
    assert "era5_codec_artifact_ready 0.0" in text
    assert "era5_codec_actual_compression_ratio" not in text
```

- [ ] **Step 2: Run the exporter tests and verify RED**

```bash
python -m pytest tests/test_ml_metrics_exporter.py -q
```

Expected: `MlArtifactCollector` is missing.

- [ ] **Step 3: Implement collector and server entry point**

`MlArtifactCollector.collect()` creates `GaugeMetricFamily` objects from a
fresh validated snapshot. Use fixed metric descriptions and only the canonical
`channel` label for per-channel series. Track the artifact modification key so
the validation-failure counter advances once per distinct invalid bundle.

The module entry point reads:

```python
root = Path(os.environ.get("ERA5_ML_METRICS_ROOT", "/workspace/artifacts/model-n32"))
host = os.environ.get("ERA5_ML_EXPORTER_HOST", "0.0.0.0")
port = int(os.environ.get("ERA5_ML_EXPORTER_PORT", "9101"))
```

Register the collector in an isolated `CollectorRegistry`, call
`start_http_server(port, addr=host, registry=registry)`, log the selected root,
and keep the process alive without reading credentials.

- [ ] **Step 4: Run exporter tests and module smoke**

```bash
python -m pytest tests/test_ml_metrics_exporter.py -q
ERA5_ML_METRICS_ROOT=artifacts/model-n32 \
  python -m era5_minimum.monitoring.ml_exporter --check
```

Expected: tests pass and `--check` reports a valid snapshot with 28 channels
without printing artifact contents or checkpoint paths.

- [ ] **Step 5: Commit**

```bash
git add src/era5_minimum/monitoring/ml_exporter.py tests/test_ml_metrics_exporter.py
git commit -m "feat: expose validated ML Prometheus metrics"
```

### Task 3: Add exporter, scrape job, and safe local Grafana access

**Files:**
- Modify: `compose.monitoring.yaml`
- Modify: `monitoring/prometheus/prometheus.yml`
- Modify: `.env.example`
- Create: `tests/test_monitoring_configuration.py`

- [ ] **Step 1: Write failing configuration tests**

Parse YAML and assert:

```python
def test_prometheus_scrapes_ml_exporter():
    config = yaml.safe_load(Path("monitoring/prometheus/prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["era5-codec-artifacts"]["static_configs"][0]["targets"] == [
        "ml-metrics-exporter:9101"
    ]


def test_compose_mounts_model_artifacts_read_only():
    compose = yaml.safe_load(Path("compose.monitoring.yaml").read_text())
    exporter = compose["services"]["ml-metrics-exporter"]
    assert exporter["environment"]["ERA5_ML_METRICS_ROOT"] == \
        "/workspace/artifacts/model-n32"
    assert any(":ro" in volume for volume in exporter["volumes"])
```

Also assert Grafana anonymous role is `Viewer`, sign-up is disabled, and host
ports use a loopback bind by default.

- [ ] **Step 2: Run configuration tests and verify RED**

```bash
python -m pytest tests/test_monitoring_configuration.py -q
```

Expected: exporter service and scrape job are absent.

- [ ] **Step 3: Implement Compose and Prometheus wiring**

Add `ml-metrics-exporter` using the existing CPU image:

```yaml
ml-metrics-exporter:
  image: era5-minimum:cpu
  command: [python, -m, era5_minimum.monitoring.ml_exporter]
  environment:
    ERA5_ML_METRICS_ROOT: /workspace/artifacts/model-n32
  ports:
    - "${ML_EXPORTER_BIND_ADDRESS:-127.0.0.1}:${ML_EXPORTER_PORT:-9101}:9101"
  volumes:
    - ${ARTIFACTS_DIR:-./artifacts}:/workspace/artifacts:ro,Z
```

Add a Python-urllib health check against `/metrics`, make Prometheus depend on
it, and add the `era5-codec-artifacts` scrape job. Bind Prometheus and Grafana
to `${..._BIND_ADDRESS:-127.0.0.1}`. Enable anonymous Grafana Viewer access,
keep sign-up disabled, and do not expose admin credentials.

Add `ML_EXPORTER_PORT`, bind-address variables, and monitoring port defaults to
`.env.example`.

- [ ] **Step 4: Verify configuration**

```bash
python -m pytest tests/test_monitoring_configuration.py -q
make monitoring-config
make monitoring-check-prometheus-config
make monitoring-check-rules
```

Expected: tests and all configuration checks pass.

- [ ] **Step 5: Commit**

```bash
git add compose.monitoring.yaml monitoring/prometheus/prometheus.yml \
  .env.example tests/test_monitoring_configuration.py
git commit -m "feat: add ML metrics exporter to monitoring stack"
```

### Task 4: Provision the hackathon model dashboard

**Files:**
- Create: `monitoring/grafana/dashboards/era5-model-overview.json`
- Modify: `tests/test_monitoring_configuration.py`

- [ ] **Step 1: Write failing dashboard contract tests**

```python
def test_model_dashboard_has_required_panels_and_honest_queries():
    dashboard = json.loads(
        Path("monitoring/grafana/dashboards/era5-model-overview.json").read_text()
    )
    assert dashboard["uid"] == "era5-model-overview"
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert {
        "Serialized compression ratio",
        "Tensor element ratio",
        "Exact quantized-symbol roundtrip",
        "Overall / surface / pressure NRMSE",
        "NRMSE by canonical channel",
        "PSNR by canonical channel",
    } <= titles
    serialized = next(
        panel for panel in dashboard["panels"]
        if panel["title"] == "Serialized compression ratio"
    )
    assert serialized["targets"][0]["expr"] == \
        "era5_codec_actual_compression_ratio"
```

Assert every query uses a metric declared by the exporter and that the text
panel explicitly marks unavailable CI/spectral/extreme/probe evidence.

- [ ] **Step 2: Run dashboard tests and verify RED**

```bash
python -m pytest tests/test_monitoring_configuration.py -q
```

Expected: dashboard file is missing.

- [ ] **Step 3: Create the dashboard JSON**

Provision UID `era5-model-overview`, title
`ERA5 Model — Compression & Quality`, datasource UID `prometheus`, refresh
`10s`, and the six rows from the design:

1. readiness, serialized ratio, tensor ratio, roundtrip, bytes, frames;
2. grouped NRMSE, PSNR, train/test sample counts;
3. canonical-channel NRMSE and PSNR bar gauges;
4. MSLP/TP6H/wind diagnostics and codec timings;
5. steps, parameters, runtime, VRAM, and GPU hours;
6. a non-inferiority evidence limitation text panel.

Use Grafana units `bytes`, `s`, `decbytes`, `percentunit`, and `dB` only where
semantically correct. Set explicit `No data` text. Do not add reference
threshold pass/fail panels without reference artifacts.

- [ ] **Step 4: Run dashboard tests**

```bash
python -m pytest tests/test_monitoring_configuration.py -q
python -m json.tool monitoring/grafana/dashboards/era5-model-overview.json \
  >/dev/null
```

Expected: all tests pass and JSON is valid.

- [ ] **Step 5: Commit**

```bash
git add monitoring/grafana/dashboards/era5-model-overview.json \
  tests/test_monitoring_configuration.py
git commit -m "feat: provision ERA5 model Grafana dashboard"
```

### Task 5: Link the frontend and extend monitoring smoke verification

**Files:**
- Modify: `src/app/settings.ts`
- Modify: `compose.yaml`
- Modify: `.env.example`
- Modify: `scripts/smoke_monitoring.py`
- Modify: `docs/MONITORING.md`
- Modify: `README.md`
- Test: `src/features/codec/api.test.ts`
- Test: `tests/test_monitoring_configuration.py`

- [ ] **Step 1: Write failing URL and smoke-contract tests**

Add a frontend assertion that
`DEFAULT_APP_SETTINGS.services.grafanaUrl` ends with the model-dashboard path.
Add Python source/config tests that the smoke script checks job
`era5-codec-artifacts`, metric `era5_codec_artifact_ready`, dashboard UID, and
the direct dashboard URL.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
npm test -- --run src/features/codec/api.test.ts
python -m pytest tests/test_monitoring_configuration.py -q
```

Expected: the Grafana default still points to the root and smoke checks are
missing.

- [ ] **Step 3: Point the frontend at the provisioned dashboard**

Set the default and Compose value to:

```text
http://localhost:3000/d/era5-model-overview/era5-model-compression-quality
```

Keep `VITE_GRAFANA_URL` overridable.

- [ ] **Step 4: Extend the smoke script**

Add `--ml-exporter-url` defaulting to `http://localhost:9101`. Verify:

- exporter `/metrics` contains `era5_codec_artifact_ready`;
- Prometheus has an up target for `era5-codec-artifacts`;
- PromQL returns readiness `1`;
- Grafana API returns dashboard UID `era5-model-overview`;
- anonymous GET of the direct dashboard URL returns 200.

Update success output to name API, exporter, Prometheus, and Grafana.

- [ ] **Step 5: Update documentation**

Document the direct dashboard URL, selected artifact root, two dashboards,
anonymous loopback-only access, exporter failure behavior, and exact commands.
Revise the old “ML metrics are deferred” section: confirmed artifact metrics
are now exported, while missing CI/spectral/extreme/probe evidence remains
deferred.

- [ ] **Step 6: Run focused verification**

```bash
npm test -- --run
npm run typecheck
python -m pytest tests/test_ml_metrics_exporter.py \
  tests/test_monitoring_configuration.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/app/settings.ts compose.yaml .env.example \
  scripts/smoke_monitoring.py docs/MONITORING.md README.md \
  src/features/codec/api.test.ts tests/test_monitoring_configuration.py
git commit -m "feat: connect frontend to Grafana model dashboard"
```

### Task 6: Full verification and local stack launch

**Files:**
- No source changes expected.

- [ ] **Step 1: Run repository verification**

```bash
make verify
npm test -- --run
npm run typecheck
npm run build
```

Expected: Python verification, all frontend tests, typecheck, and production
build pass.

- [ ] **Step 2: Validate monitoring assets**

```bash
make monitoring-config
make monitoring-check-prometheus-config
make monitoring-check-rules
```

Expected: Compose, Prometheus, and rules are valid.

- [ ] **Step 3: Stop the standalone Vite process**

Stop only the known local Vite process on port 5173 so Compose can bind the
frontend port. Do not kill unrelated Node processes.

- [ ] **Step 4: Start and smoke-test the stack**

```bash
make monitoring-up
make monitoring-ps
make monitoring-smoke
```

Expected: frontend, API, exporter, Prometheus, and Grafana are healthy.

- [ ] **Step 5: Verify the user path**

Open:

```text
http://127.0.0.1:5173/
```

Use the top-header Grafana button and confirm it opens:

```text
http://localhost:3000/d/era5-model-overview/era5-model-compression-quality
```

Capture desktop and mobile screenshots. Confirm the model dashboard shows the
real N=32 artifact values, missing optional evidence as `No data`, and no
authentication prompt.

- [ ] **Step 6: Inspect final repository state**

```bash
git status --short
git log --oneline --decorate -8
```

Expected: clean feature branch with all implementation commits and no data,
checkpoint, bitstream, generated output, credential, or monitoring-volume
changes.
