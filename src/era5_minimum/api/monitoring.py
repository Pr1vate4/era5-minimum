"""Prometheus metrics and ASGI middleware for the Artifact API."""

from __future__ import annotations

from time import perf_counter
from typing import Literal

from prometheus_client import Counter, Gauge, Histogram
from starlette.routing import Match, Router

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
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "era5_api_http_requests_in_progress",
    "HTTP requests currently being handled by the ERA5 Artifact API.",
    labelnames=("method", "route"),
)
ARTIFACT_LOAD_ERRORS_TOTAL = Counter(
    "era5_api_artifact_load_errors_total",
    "Artifact read errors by bounded artifact type.",
    labelnames=("artifact_type",),
)
ARTIFACT_VALIDATION_ERRORS_TOTAL = Counter(
    "era5_api_artifact_validation_errors_total",
    "Artifact validation errors by bounded artifact type.",
    labelnames=("artifact_type",),
)
ARTIFACTS_LOADED_TOTAL = Counter(
    "era5_api_artifacts_loaded_total",
    "Successfully loaded and validated artifacts by bounded artifact type.",
    labelnames=("artifact_type",),
)


def initialize_health_metrics() -> None:
    """Create zero-valued health metric series for an immediately useful scrape."""
    labels = {"method": "GET", "route": "/health"}
    HTTP_REQUESTS_TOTAL.labels(**labels, status_code="200")
    HTTP_REQUEST_DURATION_SECONDS.labels(**labels)
    HTTP_REQUESTS_IN_PROGRESS.labels(**labels)


def record_artifact_load_error(artifact_type: ArtifactType) -> None:
    """Record one unsuccessful JSON artifact read."""
    ARTIFACT_LOAD_ERRORS_TOTAL.labels(artifact_type=artifact_type).inc()


def record_artifact_validation_error(artifact_type: ArtifactType) -> None:
    """Record one artifact that failed repository validation."""
    ARTIFACT_VALIDATION_ERRORS_TOTAL.labels(artifact_type=artifact_type).inc()


def record_artifact_loaded(artifact_type: ArtifactType) -> None:
    """Record one successful JSON read and Pydantic validation."""
    ARTIFACTS_LOADED_TOTAL.labels(artifact_type=artifact_type).inc()


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
        HTTP_REQUESTS_IN_PROGRESS.labels(**labels).inc()

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
            HTTP_REQUESTS_IN_PROGRESS.labels(**labels).dec()
