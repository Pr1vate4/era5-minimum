"""Prometheus metrics and ASGI middleware for the Artifact API."""

from __future__ import annotations

import os
import resource
import sys
from threading import Lock
from time import perf_counter, process_time, time
from typing import Literal

from prometheus_client import (
    REGISTRY,
    PROCESS_COLLECTOR,
    Counter,
    Gauge,
    Histogram,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from starlette.routing import Match, Router

from era5_minimum import __version__

ArtifactType = Literal[
    "summary", "experiments", "sample_efficiency", "reconstruction"
]
ARTIFACT_TYPES: tuple[ArtifactType, ...] = (
    "summary",
    "experiments",
    "sample_efficiency",
    "reconstruction",
)

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
ARTIFACTS_LOADED = Gauge(
    "era5_api_artifacts_loaded",
    "Current count of artifact types successfully loaded and validated.",
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
CODEC_JOBS_TOTAL = Counter(
    "era5_api_codec_jobs_total",
    "Completed interactive codec jobs by bounded outcome.",
    labelnames=("status",),
)
CODEC_JOB_DURATION_SECONDS = Histogram(
    "era5_api_codec_job_duration_seconds",
    "End-to-end interactive codec job duration.",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

_artifact_load_states: dict[ArtifactType, int] = {
    artifact_type: 0 for artifact_type in ARTIFACT_TYPES
}
_artifact_load_states_lock = Lock()
_process_fallback_registered = False
_process_fallback_lock = Lock()
_process_start_time = time()


class _ProcessMetricsFallback:
    """Expose basic process metrics on platforms without a readable ``/proc``."""

    def collect(self):
        """Yield the process metrics expected by the monitoring contract."""
        yield CounterMetricFamily(
            "process_cpu_seconds",
            "Total user and system CPU time spent in seconds.",
            value=process_time(),
        )
        resident_memory = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            resident_memory *= 1024.0
        yield GaugeMetricFamily(
            "process_resident_memory_bytes",
            "Resident memory size in bytes.",
            value=resident_memory,
        )
        yield GaugeMetricFamily(
            "process_start_time_seconds",
            "Start time of the process since unix epoch in seconds.",
            value=_process_start_time,
        )
        try:
            open_fds = len(os.listdir("/dev/fd"))
        except OSError:
            open_fds = 0
        yield GaugeMetricFamily(
            "process_open_fds",
            "Number of open file descriptors.",
            value=open_fds,
        )


def _ensure_process_metrics() -> None:
    """Register a portable process collector only when the default is empty."""
    global _process_fallback_registered
    if _process_fallback_registered:
        return
    with _process_fallback_lock:
        if _process_fallback_registered:
            return
        if not list(PROCESS_COLLECTOR.collect()):
            REGISTRY.register(_ProcessMetricsFallback())
        _process_fallback_registered = True


def initialize_monitoring_metrics() -> None:
    """Create stable startup series without registering duplicate collectors."""
    _ensure_process_metrics()
    labels = {"method": "GET", "route": "/health"}
    HTTP_REQUESTS_TOTAL.labels(**labels, status_code="200")
    HTTP_REQUEST_DURATION_SECONDS.labels(**labels)
    HTTP_REQUESTS_IN_PROGRESS.labels(method="GET")
    ARTIFACTS_LOADED.set(sum(_artifact_load_states.values()))
    BUILD_INFO.labels(
        version=os.getenv("ERA5_BUILD_VERSION", __version__),
        commit=os.getenv("ERA5_BUILD_COMMIT", "unknown"),
        environment=os.getenv("ERA5_ENVIRONMENT", "local"),
    ).set(1)
    CODEC_JOBS_TOTAL.labels(status="success")
    CODEC_JOBS_TOTAL.labels(status="error")


def record_codec_job(*, status: Literal["success", "error"], elapsed_seconds: float | None = None) -> None:
    """Record one interactive codec request without unbounded user labels."""

    CODEC_JOBS_TOTAL.labels(status=status).inc()
    if elapsed_seconds is not None:
        CODEC_JOB_DURATION_SECONDS.observe(elapsed_seconds)


def record_artifact_load_error(artifact_type: ArtifactType) -> None:
    """Record one unsuccessful JSON artifact read."""
    ARTIFACT_LOAD_TOTAL.labels(artifact_type=artifact_type, status="error").inc()
    _set_artifact_loaded_state(artifact_type, is_loaded=False)


def record_artifact_validation_error(artifact_type: ArtifactType) -> None:
    """Record one artifact that failed repository validation."""
    ARTIFACT_VALIDATION_TOTAL.labels(
        artifact_type=artifact_type, status="invalid"
    ).inc()
    _set_artifact_loaded_state(artifact_type, is_loaded=False)


def record_artifact_loaded(artifact_type: ArtifactType) -> None:
    """Record one successful JSON read and Pydantic validation."""
    ARTIFACT_LOAD_TOTAL.labels(artifact_type=artifact_type, status="success").inc()
    ARTIFACT_VALIDATION_TOTAL.labels(
        artifact_type=artifact_type, status="valid"
    ).inc()
    _set_artifact_loaded_state(artifact_type, is_loaded=True)
    ARTIFACT_LAST_SUCCESS_TIMESTAMP_SECONDS.labels(
        artifact_type=artifact_type
    ).set(time())


def _set_artifact_loaded_state(
    artifact_type: ArtifactType, *, is_loaded: bool
) -> None:
    """Set per-type state and the bounded aggregate loaded-artifact count."""
    with _artifact_load_states_lock:
        _artifact_load_states[artifact_type] = int(is_loaded)
        ARTIFACT_LOADED.labels(artifact_type=artifact_type).set(int(is_loaded))
        ARTIFACTS_LOADED.set(sum(_artifact_load_states.values()))


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
