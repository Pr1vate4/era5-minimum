"""Contract tests for the public Prometheus endpoint."""

from fastapi.testclient import TestClient

from era5_minimum.api.app import app
from era5_minimum.api.monitoring import HTTP_REQUESTS_TOTAL


def _request_total() -> float:
    return sum(
        float(sample.value)
        for metric in HTTP_REQUESTS_TOTAL.collect()
        for sample in metric.samples
        if sample.name == "era5_api_http_requests_total"
    )


def test_metrics_endpoint_exposes_process_and_build_metrics() -> None:
    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "process_cpu_seconds_total" in response.text
    assert "era5_api_build_info" in response.text


def test_metrics_scrape_does_not_count_as_an_http_request() -> None:
    client = TestClient(app)
    before = _request_total()

    assert client.get("/metrics").status_code == 200

    assert _request_total() == before
