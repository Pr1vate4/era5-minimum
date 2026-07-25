"""Verify the local API, ML exporter, Prometheus, and Grafana stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def get(url: str, *, timeout: float = 3.0) -> tuple[int, str]:
    """Return a small HTTP response without adding a third-party dependency."""
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - local URLs only.
            return response.status, response.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def wait_for(description: str, check: Callable[[], Any], timeout: float) -> Any:
    """Retry ``check`` until it succeeds or provide a useful timeout error."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return check()
        except (AssertionError, OSError, URLError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {description}: {last_error}")


def require_ok(url: str, expected_body: str | None = None) -> str:
    """Require an HTTP 200 response, optionally containing a marker."""
    status, body = get(url)
    assert status == 200, f"{url} returned {status}"
    if expected_body is not None:
        assert expected_body in body, f"{url} did not contain {expected_body!r}"
    return body


def prometheus_query(prometheus_url: str, query: str) -> dict[str, Any]:
    """Run one instant PromQL query and return its successful JSON payload."""
    url = f"{prometheus_url}/api/v1/query?{urlencode({'query': query})}"
    status, body = get(url)
    assert status == 200, f"Prometheus query returned {status}"
    payload = json.loads(body)
    assert payload.get("status") == "success", payload
    return payload["data"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url", default=os.getenv("ERA5_API_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("ERA5_PROMETHEUS_URL", "http://localhost:9090"),
    )
    parser.add_argument(
        "--ml-exporter-url",
        default=os.getenv("ERA5_ML_EXPORTER_URL", "http://localhost:9101"),
    )
    parser.add_argument(
        "--grafana-url",
        default=os.getenv("ERA5_GRAFANA_URL", "http://localhost:3000"),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    wait_for("API health", lambda: require_ok(f"{args.api_url}/health"), args.timeout)
    require_ok(f"{args.api_url}/api/v1/summary")
    status, _ = get(f"{args.api_url}/not-a-route")
    assert status == 404, f"unmatched API route returned {status}"
    require_ok(f"{args.api_url}/metrics", "era5_api_http_requests_total")
    require_ok(
        f"{args.ml_exporter_url}/metrics",
        "era5_codec_artifact_ready 1.0",
    )
    wait_for(
        "Prometheus readiness",
        lambda: require_ok(f"{args.prometheus_url}/-/ready"),
        args.timeout,
    )

    def check_targets() -> None:
        status, body = get(f"{args.prometheus_url}/api/v1/targets")
        assert status == 200
        targets = json.loads(body)["data"]["activeTargets"]
        for job in ("era5-api", "era5-codec-artifacts"):
            assert any(
                target.get("labels", {}).get("job") == job
                and target.get("health") == "up"
                for target in targets
            ), targets

    wait_for("Prometheus scrape targets", check_targets, args.timeout)

    def check_up() -> None:
        result = prometheus_query(args.prometheus_url, 'up{job="era5-api"}')
        assert result["result"] and result["result"][0]["value"][1] == "1", result

    wait_for("Prometheus up metric", check_up, args.timeout)
    payload = prometheus_query(args.prometheus_url, "sum(era5_api_http_requests_total)")
    assert payload["result"], "HTTP request metric is absent from Prometheus"

    def check_artifact_ready() -> None:
        result = prometheus_query(
            args.prometheus_url, "era5_codec_artifact_ready"
        )
        assert result["result"] and result["result"][0]["value"][1] == "1", result

    wait_for("validated ML artifact metric", check_artifact_ready, args.timeout)
    grafana_root = args.grafana_url.rstrip("/")
    wait_for(
        "Grafana health", lambda: require_ok(f"{grafana_root}/api/health"), args.timeout
    )
    dashboard_path = "/d/era5-model-overview/era5-model-compression-quality"
    wait_for(
        "provisioned model dashboard",
        lambda: require_ok(f"{grafana_root}/api/dashboards/uid/era5-model-overview"),
        args.timeout,
    )
    wait_for(
        "anonymous model dashboard",
        lambda: require_ok(f"{grafana_root}{dashboard_path}"),
        args.timeout,
    )
    print(
        "Monitoring smoke test passed: API, ML exporter, Prometheus, "
        "and Grafana model dashboard are healthy."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, RuntimeError, URLError, ValueError) as exc:
        print(f"Monitoring smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
