"""FastAPI application for the ERA5-Minimum Artifact API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from functools import lru_cache

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
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
    initialize_monitoring_metrics,
)
from era5_minimum.api.repository import ArtifactRepository
from era5_minimum.api.weather import WeatherDataProvider, WeatherProviderError, get_weather_provider
from era5_minimum.api.schemas import (
    ErrorResponse,
    ExperimentArtifact,
    ReconstructionArtifact,
    SampleEfficiencyArtifact,
    SummaryArtifact,
)
from era5_minimum.api.weather_data_service import WeatherDataService
from era5_minimum.api.codec_service import CodecInputError, CodecService, CodecUnavailableError

DEFAULT_ARTIFACTS_ROOT = Path(
    os.getenv("ERA5_ARTIFACTS_ROOT", "demo/mock")
).resolve()


def get_repository() -> ArtifactRepository:
    """Return the default artifact repository dependency."""
    return ArtifactRepository(artifacts_root=DEFAULT_ARTIFACTS_ROOT)


@lru_cache(maxsize=1)
def get_codec_service() -> CodecService:
    """Return the process-local immutable N32 codec service."""
    return CodecService()


router = APIRouter()
metrics_asgi_app = make_asgi_app()


@lru_cache(maxsize=1)
def get_weather_data_service() -> WeatherDataService:
    return WeatherDataService(get_weather_provider())


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return the API liveness response."""
    return {"status": "ok"}


@router.get("/api/v1/codec/status", tags=["codec"])
def codec_status(service: CodecService = Depends(get_codec_service)) -> dict:
    """Expose N32 readiness for the interactive frontend without fake results."""
    return service.status()


@router.post("/api/v1/codec/jobs", status_code=201, tags=["codec"])
async def create_codec_job(
    file: UploadFile = File(...),
    target_ratio: int = Form(...),
    service: CodecService = Depends(get_codec_service),
) -> dict:
    """Compress one canonical NPZ frame with the accepted model checkpoint."""
    try:
        payload = await file.read()
        return service.create_job(
            payload=payload,
            filename=file.filename or "upload.npz",
            target_ratio=target_ratio,
        )
    except CodecUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CodecInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


def _codec_job_or_404(identifier: str, service: CodecService):
    job = service.get_job(identifier)
    if job is None:
        raise HTTPException(status_code=404, detail="codec job does not exist or its downloads expired")
    return job


@router.get("/api/v1/codec/jobs/{identifier}", tags=["codec"])
def read_codec_job(identifier: str, service: CodecService = Depends(get_codec_service)) -> dict:
    """Return a completed job during the API process lifetime."""
    return _codec_job_or_404(identifier, service).payload


@router.get("/api/v1/codec/jobs/{identifier}/bitstream", tags=["codec"])
def download_codec_bitstream(identifier: str, service: CodecService = Depends(get_codec_service)) -> Response:
    """Download the self-describing, checkpoint-bound bitstream."""
    job = _codec_job_or_404(identifier, service)
    return Response(job.artifact.bitstream, media_type="application/octet-stream", headers={"Content-Disposition": 'attachment; filename="era5-frame.e5ac"'})


@router.get("/api/v1/codec/jobs/{identifier}/reconstruction", tags=["codec"])
def download_codec_reconstruction(identifier: str, service: CodecService = Depends(get_codec_service)) -> Response:
    """Download the reconstructed canonical NPZ frame."""
    job = _codec_job_or_404(identifier, service)
    return Response(job.artifact.reconstruction, media_type="application/x-npz", headers={"Content-Disposition": 'attachment; filename="era5-reconstruction.npz"'})


@router.get("/api/v1/codec/jobs/{identifier}/preview/{kind}.png", tags=["codec"])
def codec_preview(identifier: str, kind: str, service: CodecService = Depends(get_codec_service)) -> Response:
    """Return a t2m preview for the original or reconstructed physical field."""
    job = _codec_job_or_404(identifier, service)
    if kind == "original":
        image = job.artifact.original_preview
    elif kind == "reconstruction":
        image = job.artifact.reconstruction_preview
    else:
        raise HTTPException(status_code=404, detail="preview kind must be original or reconstruction")
    return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


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


@router.get("/api/v1/datasets/current", tags=["weather"])
def current_dataset(provider: WeatherDataProvider = Depends(get_weather_provider)) -> dict:
    try:
        return provider.dataset_metadata()
    except WeatherProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/v1/variables", tags=["weather"])
def weather_variables(provider: WeatherDataProvider = Depends(get_weather_provider)) -> list[dict]:
    try:
        return provider.variables()
    except WeatherProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/v1/timestamps", tags=["weather"])
def weather_timestamps(provider: WeatherDataProvider = Depends(get_weather_provider)) -> list[str]:
    try:
        return provider.timestamps()
    except WeatherProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/v1/layers", tags=["weather"])
def weather_layer(
    variable: str,
    timestamp: str,
    response: Response,
    mode: str = Query("original", pattern="^(original|reconstructed|error)$"),
    level: int | None = None,
    target_width: int = Query(360, ge=1, le=1440),
    target_height: int = Query(180, ge=1, le=721),
    service: WeatherDataService = Depends(get_weather_data_service),
) -> dict:
    if mode != "original":
        raise HTTPException(
            status_code=501,
            detail="model reconstruction is not connected to the real Zarr provider",
        )

    try:
        payload, cache_status = service.get_layer_with_status(
            variable=variable,
            timestamp=timestamp,
            level=level,
            mode=mode,
            target_width=target_width,
            target_height=target_height,
            stride=1,
            response_format="json",
        )
    except WeatherProviderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if service.settings.enabled:
        response.headers["X-ERA5-Cache"] = cache_status.upper()

    return payload


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
    # The Vite development server is deliberately separate from the API process.
    # Keep its origins explicit rather than allowing arbitrary browser clients.
    allowed_origins = os.getenv(
        "ERA5_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
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
    initialize_monitoring_metrics()
    return api


# Keep the existing ASGI import contract: uvicorn era5_minimum.api.app:app.
app = create_app()
