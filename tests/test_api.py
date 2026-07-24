"""
Тесты HTTP-слоя (app.py) через FastAPI TestClient.

Никакой сети, NetCDF, CDS, реальных ERA5-файлов и ML-моделей —
только временные JSON-артефакты в tmp_path, как того требует задача
API-001. Repository для приложения переопределяется через
app.dependency_overrides, чтобы не зависеть от demo/mock/ на диске.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from era5_minimum.api.app import app, create_app, get_repository
from era5_minimum.api.monitoring import (
    ARTIFACT_LOADED,
    ARTIFACT_LOAD_TOTAL,
    ARTIFACT_VALIDATION_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
)
from era5_minimum.api.repository import ArtifactRepository

# ---------------------------------------------------------------------------
# Фикстуры: собираем минимальный валидный bundle во временном каталоге.
# ---------------------------------------------------------------------------
VALID_SUMMARY = {
    "is_demo": True,
    "experiment_count": 1,
    "completed": 1,
    "failed": 0,
    "generated_at": "2024-01-02T15:00:00Z",
}

VALID_EXPERIMENT = {
    "id": "demo-pca-001",
    "name": "Demo PCA baseline",
    "model": "pca",
    "status": "completed",
    "is_demo": True,
    "compression": {"type": "tensor_ratio", "ratio": 32},
    "training": {"sample_count": 168, "selection_strategy": "contiguous"},
    "metrics": {"normalized_rmse": 0.12, "mae": 0.08},
}

VALID_SAMPLE_EFFICIENCY = {
    "is_demo": True,
    "metric": "normalized_rmse",
    "points": [
        {"sample_count": 64, "value": 0.21},
        {"sample_count": 128, "value": 0.17},
    ],
}

VALID_RECONSTRUCTION = {
    "experiment_id": "demo-pca-001",
    "channel": "t2m",
    "units": "K",
    "timestamp": "2024-01-02T12:00:00Z",
    "latitude": [50.0, 51.0],
    "longitude": [10.0, 11.0, 12.0],
    "original": [[288.1, 288.4, 288.9], [287.6, 287.9, 288.2]],
    "reconstruction": [[288.0, 288.5, 288.8], [287.7, 287.8, 288.3]],
    "absolute_error": [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
    "is_demo": True,
}


def _metric_value(collector, name: str, **labels: str) -> float:
    """Return one labelled Prometheus sample, creating its zero value if needed."""
    collector.labels(**labels)
    for metric in collector.collect():
        for sample in metric.samples:
            if sample.name == name and sample.labels == labels:
                return float(sample.value)
    raise AssertionError(f"Prometheus sample not found: {name} {labels}")


def _http_request_total() -> float:
    """Return all user HTTP counter values without creating new labelsets."""
    return sum(
        float(sample.value)
        for metric in HTTP_REQUESTS_TOTAL.collect()
        for sample in metric.samples
        if sample.name == "era5_api_http_requests_total"
    )


def _write_bundle(root: Path, **overrides) -> None:
    """
    Пишет минимальный валидный bundle в `root`, с точечными
    переопределениями (`overrides`) для проверки ошибочных сценариев.

    Ключи overrides: "summary", "experiments", "sample_efficiency",
    "reconstruction" (dict или сырая строка для проверки malformed JSON).
    """
    (root / "reconstructions").mkdir(parents=True, exist_ok=True)

    files = {
        "summary.json": overrides.get("summary", VALID_SUMMARY),
        "experiments.json": overrides.get("experiments", VALID_EXPERIMENT),
        "sample_efficiency.json": overrides.get(
            "sample_efficiency", VALID_SAMPLE_EFFICIENCY
        ),
    }
    for name, content in files.items():
        path = root / name
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_text(json.dumps(content), encoding="utf-8")

    reconstruction = overrides.get("reconstruction", VALID_RECONSTRUCTION)
    recon_filename = overrides.get(
        "reconstruction_filename", "demo-pca-001.json"
    )
    recon_path = root / "reconstructions" / recon_filename
    if isinstance(reconstruction, str):
        recon_path.write_text(reconstruction, encoding="utf-8")
    else:
        recon_path.write_text(json.dumps(reconstruction), encoding="utf-8")


@pytest.fixture()
def client_factory(tmp_path):
    """
    Возвращает функцию, которая строит TestClient поверх bundle
    в tmp_path с заданными переопределениями.
    """

    def _make(**overrides) -> TestClient:
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        _write_bundle(bundle_dir, **overrides)

        def _override_repository() -> ArtifactRepository:
            return ArtifactRepository(artifacts_root=bundle_dir)

        app.dependency_overrides[get_repository] = _override_repository
        client = TestClient(app)
        return client

    yield _make
    app.dependency_overrides.clear()


@pytest.fixture()
def client(client_factory) -> TestClient:
    return client_factory()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Prometheus instrumentation
# ---------------------------------------------------------------------------
def test_metrics_endpoint_uses_prometheus_exposition_format(
    client: TestClient,
) -> None:
    response = client.get("/metrics", follow_redirects=False)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "era5_api_http_requests_total" in response.text
    assert "era5_api_http_request_duration_seconds_bucket" in response.text
    assert "era5_api_http_requests_in_progress" in response.text
    assert "era5_api_build_info" in response.text
    assert "process_cpu_seconds_total" in response.text
    assert "process_resident_memory_bytes" in response.text
    assert "process_open_fds" in response.text
    assert "process_start_time_seconds" in response.text


def test_health_records_request_counter_and_duration(client: TestClient) -> None:
    labels = {"method": "GET", "route": "/health"}
    before_requests = _metric_value(
        HTTP_REQUESTS_TOTAL,
        "era5_api_http_requests_total",
        **labels,
        status_code="200",
    )
    before_duration_count = _metric_value(
        HTTP_REQUEST_DURATION_SECONDS,
        "era5_api_http_request_duration_seconds_count",
        **labels,
    )

    assert client.get("/health").status_code == 200

    assert _metric_value(
        HTTP_REQUESTS_TOTAL,
        "era5_api_http_requests_total",
        **labels,
        status_code="200",
    ) == before_requests + 1
    assert _metric_value(
        HTTP_REQUEST_DURATION_SECONDS,
        "era5_api_http_request_duration_seconds_count",
        **labels,
    ) == before_duration_count + 1


def test_unknown_path_is_recorded_as_unmatched(client: TestClient) -> None:
    labels = {"method": "GET", "route": "unmatched", "status_code": "404"}
    before = _metric_value(
        HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels
    )

    assert client.get("/not-a-route").status_code == 404

    assert _metric_value(
        HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels
    ) == before + 1


def test_dynamic_route_uses_template_not_experiment_id(client: TestClient) -> None:
    labels = {
        "method": "GET",
        "route": "/api/v1/experiments/{experiment_id}",
        "status_code": "200",
    }
    before = _metric_value(
        HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels
    )

    assert client.get("/api/v1/experiments/demo-pca-001").status_code == 200

    assert _metric_value(
        HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels
    ) == before + 1


def test_metrics_scrape_is_excluded_from_user_request_counter(
    client: TestClient,
) -> None:
    before = _http_request_total()
    assert client.get("/metrics").status_code == 200
    assert _http_request_total() == before


def test_missing_artifact_increments_load_error_counter(client_factory) -> None:
    labels = {"artifact_type": "summary", "status": "error"}
    before = _metric_value(
        ARTIFACT_LOAD_TOTAL,
        "era5_api_artifact_load_total",
        **labels,
    )
    client = client_factory()
    bundle_root = client.app.dependency_overrides[get_repository]().root
    (bundle_root / "summary.json").unlink()

    assert client.get("/api/v1/summary").status_code == 500

    assert _metric_value(
        ARTIFACT_LOAD_TOTAL,
        "era5_api_artifact_load_total",
        **labels,
    ) == before + 1
    assert _metric_value(
        ARTIFACT_LOADED,
        "era5_api_artifact_loaded",
        artifact_type="summary",
    ) == 0


def test_successful_artifact_load_records_success_valid_and_loaded(client) -> None:
    load_labels = {"artifact_type": "summary", "status": "success"}
    validation_labels = {"artifact_type": "summary", "status": "valid"}
    before_load = _metric_value(
        ARTIFACT_LOAD_TOTAL, "era5_api_artifact_load_total", **load_labels
    )
    before_validation = _metric_value(
        ARTIFACT_VALIDATION_TOTAL,
        "era5_api_artifact_validation_total",
        **validation_labels,
    )

    assert client.get("/api/v1/summary").status_code == 200

    assert _metric_value(
        ARTIFACT_LOAD_TOTAL, "era5_api_artifact_load_total", **load_labels
    ) == before_load + 1
    assert _metric_value(
        ARTIFACT_VALIDATION_TOTAL,
        "era5_api_artifact_validation_total",
        **validation_labels,
    ) == before_validation + 1
    assert _metric_value(
        ARTIFACT_LOADED,
        "era5_api_artifact_loaded",
        artifact_type="summary",
    ) == 1


def test_invalid_artifact_increments_validation_error_counter(
    client_factory,
) -> None:
    labels = {"artifact_type": "experiments", "status": "invalid"}
    before = _metric_value(
        ARTIFACT_VALIDATION_TOTAL,
        "era5_api_artifact_validation_total",
        **labels,
    )
    client = client_factory(experiments={"id": "demo-pca-001"})

    assert client.get("/api/v1/experiments").status_code == 500

    assert _metric_value(
        ARTIFACT_VALIDATION_TOTAL,
        "era5_api_artifact_validation_total",
        **labels,
    ) == before + 1


def test_repeated_app_creation_reuses_metrics_collectors() -> None:
    first_app = create_app()
    second_app = create_app()

    assert TestClient(first_app).get("/health").status_code == 200
    assert TestClient(second_app).get("/health").status_code == 200


# ---------------------------------------------------------------------------
# /api/v1/summary
# ---------------------------------------------------------------------------
def test_summary_ok(client: TestClient) -> None:
    response = client.get("/api/v1/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["is_demo"] is True
    assert body["experiment_count"] == 1


def test_summary_missing_artifact(client_factory) -> None:
    """Отсутствующий artifact -> контролируемая серверная ошибка (500)."""
    client = client_factory()
    # Удаляем summary.json уже после сборки bundle.
    bundle_root = client.app.dependency_overrides[get_repository]().root
    (bundle_root / "summary.json").unlink()

    response = client.get("/api/v1/summary")
    assert response.status_code == 500
    body = response.json()
    assert "detail" in body
    assert str(bundle_root) not in body["detail"]


def test_summary_malformed_json(client_factory) -> None:
    client = client_factory(summary="{not valid json")
    response = client.get("/api/v1/summary")
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# /api/v1/experiments
# ---------------------------------------------------------------------------
def test_list_experiments(client: TestClient) -> None:
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["id"] == "demo-pca-001"


def test_get_experiment_found(client: TestClient) -> None:
    response = client.get("/api/v1/experiments/demo-pca-001")
    assert response.status_code == 200
    assert response.json()["id"] == "demo-pca-001"


def test_get_experiment_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/experiments/unknown-id")
    assert response.status_code == 404


def test_experiments_pydantic_validation_error(client_factory) -> None:
    """Неправильная структура JSON -> ошибка валидации (500)."""
    broken_experiment = {"id": "demo-pca-001", "name": "Broken"}  # без полей
    client = client_factory(experiments=broken_experiment)
    response = client.get("/api/v1/experiments")
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# /api/v1/sample-efficiency
# ---------------------------------------------------------------------------
def test_sample_efficiency(client: TestClient) -> None:
    response = client.get("/api/v1/sample-efficiency")
    assert response.status_code == 200
    body = response.json()
    assert body["metric"] == "normalized_rmse"
    assert len(body["points"]) == 2


# ---------------------------------------------------------------------------
# /api/v1/reconstructions/{experiment_id}
# ---------------------------------------------------------------------------
def test_reconstruction_ok(client: TestClient) -> None:
    response = client.get(
        "/api/v1/reconstructions/demo-pca-001"
        "?channel=t2m&timestamp=2024-01-02T12:00:00Z"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["channel"] == "t2m"
    assert body["experiment_id"] == "demo-pca-001"


def test_reconstruction_unknown_experiment(client: TestClient) -> None:
    response = client.get("/api/v1/reconstructions/unknown-id")
    assert response.status_code == 404


def test_reconstruction_invalid_channel(client: TestClient) -> None:
    response = client.get(
        "/api/v1/reconstructions/demo-pca-001?channel=mslp"
    )
    assert response.status_code == 422


def test_reconstruction_timestamp_not_found(client: TestClient) -> None:
    response = client.get(
        "/api/v1/reconstructions/demo-pca-001"
        "?timestamp=2099-01-01T00:00:00Z"
    )
    assert response.status_code == 422


def test_reconstruction_mismatched_shapes(client_factory) -> None:
    """Несовпадающие формы карт -> ошибка валидации (500)."""
    broken_reconstruction = dict(VALID_RECONSTRUCTION)
    broken_reconstruction["reconstruction"] = [[1.0, 2.0]]  # другая форма
    client = client_factory(reconstruction=broken_reconstruction)
    response = client.get("/api/v1/reconstructions/demo-pca-001")
    assert response.status_code == 500


def test_reconstruction_missing_artifact_file(client_factory) -> None:
    client = client_factory()
    bundle_root = client.app.dependency_overrides[get_repository]().root
    (bundle_root / "reconstructions" / "demo-pca-001.json").unlink()

    response = client.get("/api/v1/reconstructions/demo-pca-001")
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# OpenAPI / Swagger
# ---------------------------------------------------------------------------
def test_openapi_and_docs_available(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_openapi_lists_all_required_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    expected = {
        "/health",
        "/api/v1/summary",
        "/api/v1/experiments",
        "/api/v1/experiments/{experiment_id}",
        "/api/v1/sample-efficiency",
        "/api/v1/reconstructions/{experiment_id}",
    }
    assert expected.issubset(paths)
