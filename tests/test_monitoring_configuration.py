from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(relative_path: str) -> dict[str, object]:
    value = yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_prometheus_scrapes_ml_exporter() -> None:
    config = _read_yaml("monitoring/prometheus/prometheus.yml")
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}

    assert jobs["era5-codec-artifacts"]["static_configs"][0]["targets"] == [
        "ml-metrics-exporter:9101"
    ]


def test_compose_mounts_model_artifacts_read_only() -> None:
    compose = _read_yaml("compose.monitoring.yaml")
    exporter = compose["services"]["ml-metrics-exporter"]

    assert (
        exporter["environment"]["ERA5_ML_METRICS_ROOT"]
        == "/workspace/artifacts/model-n32"
    )
    assert any(volume.endswith(":ro,Z") for volume in exporter["volumes"])
    assert (
        exporter["ports"][0]
        == "${ML_EXPORTER_BIND_ADDRESS:-127.0.0.1}:${ML_EXPORTER_PORT:-9101}:9101"
    )


def test_monitoring_host_ports_default_to_loopback() -> None:
    compose = _read_yaml("compose.monitoring.yaml")
    services = compose["services"]

    assert (
        services["prometheus"]["ports"][0]
        == "${PROMETHEUS_BIND_ADDRESS:-127.0.0.1}:${PROMETHEUS_PORT:-9090}:9090"
    )
    assert (
        services["grafana"]["ports"][0]
        == "${GRAFANA_BIND_ADDRESS:-127.0.0.1}:${GRAFANA_PORT:-3000}:3000"
    )


def test_grafana_allows_anonymous_viewer_without_signup() -> None:
    compose = _read_yaml("compose.monitoring.yaml")
    environment = compose["services"]["grafana"]["environment"]

    assert environment["GF_AUTH_ANONYMOUS_ENABLED"] == "true"
    assert environment["GF_AUTH_ANONYMOUS_ORG_ROLE"] == "Viewer"
    assert environment["GF_USERS_ALLOW_SIGN_UP"] == "false"


def test_model_dashboard_has_required_panels_and_honest_queries() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/era5-model-overview.json").read_text(
            encoding="utf-8"
        )
    )

    assert dashboard["uid"] == "era5-model-overview"
    assert dashboard["title"] == "ERA5 Model — Compression & Quality"
    assert dashboard["refresh"] == "10s"
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert {
        "Serialized compression ratio",
        "Tensor element ratio",
        "Exact quantized-symbol roundtrip",
        "Overall / surface / pressure NRMSE",
        "NRMSE by canonical channel",
        "PSNR by canonical channel",
        "Evidence not present in this artifact",
    } <= titles
    serialized = next(
        panel
        for panel in dashboard["panels"]
        if panel["title"] == "Serialized compression ratio"
    )
    assert serialized["targets"][0]["expr"] == "era5_codec_actual_compression_ratio"


def test_model_dashboard_queries_only_exported_metrics() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/era5-model-overview.json").read_text(
            encoding="utf-8"
        )
    )
    exported = {
        "era5_codec_artifact_ready",
        "era5_codec_actual_compression_ratio",
        "era5_codec_tensor_compression_ratio",
        "era5_codec_bitstream_bytes",
        "era5_codec_bits_per_value",
        "era5_codec_exact_roundtrip",
        "era5_codec_frame_count",
        "era5_codec_overall_score",
        "era5_codec_surface_score",
        "era5_codec_pressure_score",
        "era5_codec_mean_psnr_db",
        "era5_codec_channel_nrmse",
        "era5_codec_channel_psnr_db",
        "era5_codec_mslp_rmse_hpa",
        "era5_codec_tp6h_rmse_mm_per_6h",
        "era5_codec_wind_speed_rmse_m_per_s",
        "era5_codec_encode_duration_seconds",
        "era5_codec_decode_duration_seconds",
        "era5_codec_encode_per_frame_seconds",
        "era5_codec_decode_per_frame_seconds",
        "era5_codec_unique_train_timestamps",
        "era5_codec_unique_validation_timestamps",
        "era5_codec_unique_test_timestamps",
        "era5_codec_optimizer_steps",
        "era5_codec_trainable_parameters",
        "era5_codec_training_duration_seconds",
        "era5_codec_peak_vram_bytes",
        "era5_codec_gpu_hours",
    }
    expressions = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    ]
    queried = {
        metric
        for expression in expressions
        for metric in re.findall(r"\bera5_codec_[a-z0-9_]+\b", expression)
    }

    assert queried <= exported
    limitation = next(
        panel
        for panel in dashboard["panels"]
        if panel["title"] == "Evidence not present in this artifact"
    )
    content = limitation["options"]["content"].lower()
    assert all(
        term in content
        for term in ("vaeformer", "confidence interval", "spectral", "extreme", "probe")
    )


def test_model_dashboard_uses_grafana_12_color_modes() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/era5-model-overview.json").read_text(
            encoding="utf-8"
        )
    )
    color_modes = {
        panel.get("fieldConfig", {}).get("defaults", {}).get("color", {}).get("mode")
        for panel in dashboard["panels"]
    }

    assert "fixedColor" not in color_modes


def test_frontend_compose_points_at_model_dashboard() -> None:
    compose = _read_yaml("compose.yaml")

    assert (
        compose["services"]["frontend"]["environment"]["VITE_GRAFANA_URL"]
        == "${VITE_GRAFANA_URL:-http://localhost:3000/d/era5-model-overview/"
        "era5-model-compression-quality}"
    )


def test_monitoring_smoke_covers_exporter_target_and_dashboard() -> None:
    source = (ROOT / "scripts/smoke_monitoring.py").read_text(encoding="utf-8")

    assert "--ml-exporter-url" in source
    assert "era5-codec-artifacts" in source
    assert "era5_codec_artifact_ready" in source
    assert "era5-model-overview" in source
    assert "/d/era5-model-overview/era5-model-compression-quality" in source
