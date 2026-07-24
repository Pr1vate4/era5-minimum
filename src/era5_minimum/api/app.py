"""FastAPI application for the ERA5-Minimum Artifact API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import make_asgi_app

from era5_minimum.api.errors import (
    ArtifactNotFoundError,
    ExperimentNotFoundError,
    InvalidArtifactError,
    MalformedJSONError,
    TimestampNotFoundError,
    UnsupportedChannelError,
)
from era5_minimum.api.monitoring import (
    METRICS_PATH,
    PrometheusMetricsMiddleware,
    initialize_health_metrics,
)
from era5_minimum.api.repository import ArtifactRepository
from era5_minimum.api.schemas import (
    ErrorResponse,
    ExperimentArtifact,
    ReconstructionArtifact,
    SampleEfficiencyArtifact,
    SummaryArtifact,
)

DEFAULT_ARTIFACTS_ROOT = Path(
    os.getenv("ERA5_ARTIFACTS_ROOT", "demo/mock")
).resolve()


def get_repository() -> ArtifactRepository:
    """Return the default artifact repository dependency."""
    return ArtifactRepository(artifacts_root=DEFAULT_ARTIFACTS_ROOT)


router = APIRouter()
metrics_asgi_app = make_asgi_app()


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return the API liveness response."""
    return {"status": "ok"}


@router.get(
    "/api/v1/summary",
    response_model=SummaryArtifact,
    responses={500: {"model": ErrorResponse}},
    tags=["summary"],
)
def read_summary(
    repository: ArtifactRepository = Depends(get_repository),
) -> SummaryArtifact:
    return repository.get_summary()


@router.get(
    "/api/v1/experiments",
    response_model=list[ExperimentArtifact],
    responses={500: {"model": ErrorResponse}},
    tags=["experiments"],
)
def list_experiments(
    repository: ArtifactRepository = Depends(get_repository),
) -> list[ExperimentArtifact]:
    return repository.list_experiments()


@router.get(
    "/api/v1/experiments/{experiment_id}",
    response_model=ExperimentArtifact,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["experiments"],
)
def read_experiment(
    experiment_id: str,
    repository: ArtifactRepository = Depends(get_repository),
) -> ExperimentArtifact:
    return repository.get_experiment(experiment_id)


@router.get(
    "/api/v1/sample-efficiency",
    response_model=SampleEfficiencyArtifact,
    responses={500: {"model": ErrorResponse}},
    tags=["sample-efficiency"],
)
def read_sample_efficiency(
    repository: ArtifactRepository = Depends(get_repository),
) -> SampleEfficiencyArtifact:
    return repository.get_sample_efficiency()


@router.get(
    "/api/v1/reconstructions/{experiment_id}",
    response_model=ReconstructionArtifact,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["reconstructions"],
)
def read_reconstruction(
    experiment_id: str,
    channel: Optional[str] = Query(
        default=None, description="Канонический код канала, напр. t2m"
    ),
    timestamp: Optional[str] = Query(
        default=None, description="ISO-8601 временная метка"
    ),
    repository: ArtifactRepository = Depends(get_repository),
) -> ReconstructionArtifact:
    return repository.get_reconstruction(
        experiment_id=experiment_id, channel=channel, timestamp=timestamp
    )


async def handle_experiment_not_found(
    request: Request, exc: ExperimentNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "experiment_not_found"},
    )


async def handle_unsupported_channel(
    request: Request, exc: UnsupportedChannelError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "unsupported_channel"},
    )


async def handle_timestamp_not_found(
    request: Request, exc: TimestampNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "timestamp_not_found"},
    )


async def handle_artifact_not_found(
    request: Request, exc: ArtifactNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Required artifact is missing",
            "error_type": "artifact_not_found",
        },
    )


async def handle_malformed_json(
    request: Request, exc: MalformedJSONError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Artifact contains malformed JSON",
            "error_type": "malformed_json",
        },
    )


async def handle_invalid_artifact(
    request: Request, exc: InvalidArtifactError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Artifact '{exc.artifact_name}' failed validation",
            "error_type": "invalid_artifact",
        },
    )


async def metrics(request: Request) -> Response:
    """Expose the official Prometheus ASGI adapter without a slash redirect."""
    response_start: dict[str, object] | None = None
    response_body = bytearray()

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        nonlocal response_start
        if message["type"] == "http.response.start":
            response_start = message
        elif message["type"] == "http.response.body":
            response_body.extend(bytes(message.get("body", b"")))

    await metrics_asgi_app(dict(request.scope), receive, send)
    if response_start is None:  # pragma: no cover - adapter always starts a response.
        raise RuntimeError("Prometheus ASGI adapter returned no response")

    headers = {
        bytes(name).decode("latin-1"): bytes(value).decode("latin-1")
        for name, value in response_start["headers"]
    }
    return Response(
        content=bytes(response_body),
        status_code=int(response_start["status"]),
        headers=headers,
    )


def create_app() -> FastAPI:
    """Create an API instance while sharing process-wide Prometheus collectors."""
    api = FastAPI(
        title="ERA5-Minimum Artifact API",
        version="0.1.0",
        description=(
            "Backend, отдающий ML-артефакты проекта ERA5-Minimum согласно "
            "docs/ARTIFACT_API_CONTRACT.md."
        ),
    )
    api.include_router(router)
    api.add_exception_handler(ExperimentNotFoundError, handle_experiment_not_found)
    api.add_exception_handler(UnsupportedChannelError, handle_unsupported_channel)
    api.add_exception_handler(TimestampNotFoundError, handle_timestamp_not_found)
    api.add_exception_handler(ArtifactNotFoundError, handle_artifact_not_found)
    api.add_exception_handler(MalformedJSONError, handle_malformed_json)
    api.add_exception_handler(InvalidArtifactError, handle_invalid_artifact)
    api.add_middleware(PrometheusMetricsMiddleware, router=api.router)
    api.add_api_route(METRICS_PATH, metrics, methods=["GET"], include_in_schema=False)
    initialize_health_metrics()
    return api


# Keep the existing ASGI import contract: uvicorn era5_minimum.api.app:app.
app = create_app()
