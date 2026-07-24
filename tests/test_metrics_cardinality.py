"""Regression tests that prevent unbounded Prometheus label cardinality."""

from fastapi.testclient import TestClient

from era5_minimum.api.app import app
from era5_minimum.api.monitoring import HTTP_REQUESTS_TOTAL


def _routes_for_status(status_code: str) -> set[str]:
    return {
        sample.labels["route"]
        for metric in HTTP_REQUESTS_TOTAL.collect()
        for sample in metric.samples
        if sample.name == "era5_api_http_requests_total"
        and sample.labels.get("status_code") == status_code
    }


def test_dynamic_ids_and_query_parameters_use_one_route_template() -> None:
    client = TestClient(app)
    for experiment_id in ("one", "two", "three"):
        response = client.get(
            f"/api/v1/experiments/{experiment_id}?filename={experiment_id}.json"
        )
        assert response.status_code == 404

    assert "/api/v1/experiments/{experiment_id}" in _routes_for_status("404")
    assert not any("one" in route or "two" in route for route in _routes_for_status("404"))


def test_unmatched_paths_create_one_bounded_route_label() -> None:
    client = TestClient(app)
    for path in ("/missing/a", "/missing/b", "/missing/c"):
        assert client.get(path).status_code == 404

    assert "unmatched" in _routes_for_status("404")


def test_metric_labels_cannot_contain_filename_or_exception_text() -> None:
    client = TestClient(app)
    filename = "secret-results.json"
    assert client.get(f"/api/v1/experiments/not-found?filename={filename}").status_code == 404

    labels = {
        key: value
        for metric in HTTP_REQUESTS_TOTAL.collect()
        for sample in metric.samples
        if sample.name == "era5_api_http_requests_total"
        for key, value in sample.labels.items()
    }
    assert filename not in labels.values()
    assert "exception" not in labels
