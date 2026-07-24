"""Prometheus metrics and ASGI middleware for the Artifact API."""

from __future__ import annotations

import os
from time import perf_counter, time
from typing import Literal

from prometheus_client import Counter, Gauge, Histogram
from starlette.routing import Match, Router

from era5_minimum import __version__

ArtifactType = Literal[
    "summary", "experiments", "sample_efficiency", "reconstruction"
]

METRICS_PATH = "/metrics"
UNMATCHED_ROUTE = "unmatched"

HTTP_REQUESTS_TOTAL = Counter(
    "era5_api_http_requests_total",
    "Total HTTP requests handled by the ERA5 Artifact API.",
    labelnames=("method", "route", "status_code"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "era5_api_http_request_duration_seconds",
    "HTTP request duration for the ERA5 Artifact API.",
    labelnames=("method", "route"),
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
        30,
    ),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "era5_api_http_requests_in_progress",
    "HTTP requests currently being handled by the ERA5 Artifact API.",
    labelnames=("method",),
)
ARTIFACT_LOAD_TOTAL = Counter(
    "era5_api_artifact_load_total",
    "Artifact read and validation outcomes by bounded artifact type.",
    labelnames=("artifact_type", "status"),
)
ARTIFACT_VALIDATION_TOTAL = Counter(
    "era5_api_artifact_validation_total",
    "Artifact validation outcomes by bounded artifact type.",
    labelnames=("artifact_type", "status"),
)
ARTIFACT_LOADED = Gauge(
    "era5_api_artifact_loaded",
    "Whether the last repository operation successfully loaded each artifact type.",
    labelnames=("artifact_type",),
)
ARTIFACT_LAST_SUCCESS_TIMESTAMP_SECONDS = Gauge(
    "era5_api_artifact_last_success_timestamp_seconds",
    "Unix time of the last successful artifact load.",
    labelnames=("artifact_type",),
)
BUILD_INFO = Gauge(
    "era5_api_build_info",
    "Static build information for the ERA5 Artifact API.",
    labelnames=("version", "commit", "environment"),
)


def initialize_monitoring_metrics() -> None:
    """Create stable startup series without registering duplicate collectors."""
    labels = {"method": "GET", "route": "/health"}
    HTTP_REQUESTS_TOTAL.labels(**labels, status_code="200")
    HTTP_REQUEST_DURATION_SECONDS.labels(**labels)
    HTTP_REQUESTS_IN_PROGRESS.labels(method="GET")
    BUILD_INFO.labels(
        version=os.getenv("ERA5_BUILD_VERSION", __version__),
        commit=os.getenv("ERA5_BUILD_COMMIT", "unknown"),
        environment=os.getenv("ERA5_ENVIRONMENT", "local"),
    ).set(1)


def record_artifact_load_error(artifact_type: ArtifactType) -> None:
    """Record one unsuccessful JSON artifact read."""
    ARTIFACT_LOAD_TOTAL.labels(artifact_type=artifact_type, status="error").inc()
    ARTIFACT_LOADED.labels(artifact_type=artifact_type).set(0)


def record_artifact_validation_error(artifact_type: ArtifactType) -> None:
    """Record one artifact that failed repository validation."""
    ARTIFACT_VALIDATION_TOTAL.labels(
        artifact_type=artifact_type, status="invalid"
    ).inc()
    ARTIFACT_LOADED.labels(artifact_type=artifact_type).set(0)


def record_artifact_loaded(artifact_type: ArtifactType) -> None:
    """Record one successful JSON read and Pydantic validation."""
    ARTIFACT_LOAD_TOTAL.labels(artifact_type=artifact_type, status="success").inc()
    ARTIFACT_VALIDATION_TOTAL.labels(
        artifact_type=artifact_type, status="valid"
    ).inc()
    ARTIFACT_LOADED.labels(artifact_type=artifact_type).set(1)
    ARTIFACT_LAST_SUCCESS_TIMESTAMP_SECONDS.labels(
        artifact_type=artifact_type
    ).set(time())


def normalized_route(scope: dict[str, object], router: Router) -> str:
    """Return the matched FastAPI route template without dynamic values."""
    for route in router.routes:
        match, _ = route.matches(scope)
        if match is Match.FULL:
            included_router = getattr(route, "original_router", None)
            if included_router is not None:
                return normalized_route(scope, included_router)
            return str(
                getattr(route, "path_format", getattr(route, "path", UNMATCHED_ROUTE))
            )
    return UNMATCHED_ROUTE


class PrometheusMetricsMiddleware:
    """Collect bounded-label HTTP metrics without changing API responses."""

    def __init__(self, app: object, *, router: Router) -> None:
        self.app = app
        self.router = router

    async def __call__(
        self, scope: dict[str, object], receive: object, send: object
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope["path"])
        if path == METRICS_PATH or path.startswith(f"{METRICS_PATH}/"):
            await self.app(scope, receive, send)
            return

        method = str(scope["method"])
        route = normalized_route(scope, self.router)
        status_code = 500
        started_at = perf_counter()
        labels = {"method": method, "route": route}
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method).inc()

        async def send_with_status(message: dict[str, object]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        except Exception:
            status_code = 500
            HTTP_REQUESTS_TOTAL.labels(
                **labels, status_code=str(status_code)
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(**labels).observe(
                perf_counter() - started_at
            )
            raise
        else:
            HTTP_REQUESTS_TOTAL.labels(
                **labels, status_code=str(status_code)
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(**labels).observe(
                perf_counter() - started_at
            )
        finally:
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method).dec()
