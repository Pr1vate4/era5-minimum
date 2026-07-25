from __future__ import annotations

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
