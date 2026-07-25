"""HTTP middleware metric behavior, including bounded failure accounting."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from era5_minimum.api.app import app
from era5_minimum.api.monitoring import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
    PrometheusMetricsMiddleware,
)


def _value(collector, sample_name: str, **labels: str) -> float:
    collector.labels(**labels)
    for metric in collector.collect():
        for sample in metric.samples:
            if sample.name == sample_name and sample.labels == labels:
                return float(sample.value)
    raise AssertionError(f"missing sample {sample_name} {labels}")


def test_health_updates_counter_duration_and_in_progress_gauge() -> None:
    client = TestClient(app)
    labels = {"method": "GET", "route": "/health"}
    before_count = _value(
        HTTP_REQUESTS_TOTAL,
        "era5_api_http_requests_total",
        **labels,
        status_code="200",
    )
    before_duration = _value(
        HTTP_REQUEST_DURATION_SECONDS,
        "era5_api_http_request_duration_seconds_count",
        **labels,
    )

    assert client.get("/health").status_code == 200

    assert _value(
        HTTP_REQUESTS_TOTAL,
        "era5_api_http_requests_total",
        **labels,
        status_code="200",
    ) == before_count + 1
    assert _value(
        HTTP_REQUEST_DURATION_SECONDS,
        "era5_api_http_request_duration_seconds_count",
        **labels,
    ) == before_duration + 1
    assert _value(
        HTTP_REQUESTS_IN_PROGRESS,
        "era5_api_http_requests_in_progress",
        method="GET",
    ) == 0


def test_unmatched_route_is_recorded_as_404() -> None:
    client = TestClient(app)
    labels = {"method": "GET", "route": "unmatched", "status_code": "404"}
    before = _value(HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels)

    assert client.get("/does-not-exist").status_code == 404

    assert _value(HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels) == before + 1


def test_unhandled_exception_is_counted_and_not_suppressed() -> None:
    failing_app = FastAPI()

    @failing_app.get("/boom")
    def boom() -> None:
        raise RuntimeError("not a metric label")

    failing_app.add_middleware(PrometheusMetricsMiddleware, router=failing_app.router)
    labels = {"method": "GET", "route": "/boom", "status_code": "500"}
    before = _value(HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels)

    client = TestClient(failing_app, raise_server_exceptions=False)
    assert client.get("/boom").status_code == 500
    assert _value(HTTP_REQUESTS_TOTAL, "era5_api_http_requests_total", **labels) == before + 1

    raising_client = TestClient(failing_app, raise_server_exceptions=True)
    try:
        raising_client.get("/boom")
    except RuntimeError as exc:
        assert str(exc) == "not a metric label"
    else:  # pragma: no cover - protects the error propagation contract.
        raise AssertionError("unhandled exception was suppressed")
