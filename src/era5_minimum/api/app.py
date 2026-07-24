"""
FastAPI-приложение ERA5-Minimum Artifact API.

Реализует все endpoint'ы, требуемые задачей API-001:

    GET /health
    GET /api/v1/summary
    GET /api/v1/experiments
    GET /api/v1/experiments/{experiment_id}
    GET /api/v1/sample-efficiency
    GET /api/v1/reconstructions/{experiment_id}?channel=&timestamp=

Здесь же — единственное место, где доменные исключения из errors.py
превращаются в HTTP-ответы:

    404  ExperimentNotFoundError
    422  UnsupportedChannelError
    422  TimestampNotFoundError
    500  ArtifactNotFoundError   (отсутствующий обязательный артефакт)
    500  MalformedJSONError      (битый JSON)
    500  InvalidArtifactError    (не прошёл Pydantic-валидацию)

Тела ответов об ошибках — только "detail"/"error_type", без traceback,
абсолютных путей, переменных окружения или токенов (контракт, раздел 12,
и требования задачи).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from era5_minimum.api.errors import (
    ArtifactNotFoundError,
    ExperimentNotFoundError,
    InvalidArtifactError,
    MalformedJSONError,
    TimestampNotFoundError,
    UnsupportedChannelError,
)
from era5_minimum.api.repository import ArtifactRepository
from era5_minimum.api.schemas import (
    ErrorResponse,
    ExperimentArtifact,
    ReconstructionArtifact,
    SampleEfficiencyArtifact,
    SummaryArtifact,
)

# Каталог с mock-артефактами по умолчанию — demo/mock из контракта.
# Переопределяется переменной окружения (например, в тестах через
# app.dependency_overrides, а не через env — см. tests/test_api.py).
DEFAULT_ARTIFACTS_ROOT = Path(
    os.getenv("ERA5_ARTIFACTS_ROOT", "demo/mock")
).resolve()


def get_repository() -> ArtifactRepository:
    """
    FastAPI-зависимость, возвращающая репозиторий артефактов.

    В тестах переопределяется через
    app.dependency_overrides[get_repository] с указанием tmp_path.
    """
    return ArtifactRepository(artifacts_root=DEFAULT_ARTIFACTS_ROOT)


app = FastAPI(
    title="ERA5-Minimum Artifact API",
    version="0.1.0",
    description=(
        "Backend, отдающий ML-артефакты проекта ERA5-Minimum согласно "
        "docs/ARTIFACT_API_CONTRACT.md."
    ),
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/api/v1/summary",
    response_model=SummaryArtifact,
    responses={500: {"model": ErrorResponse}},
    tags=["summary"],
)
def read_summary(
    repository: ArtifactRepository = Depends(get_repository),
) -> SummaryArtifact:
    return repository.get_summary()


@app.get(
    "/api/v1/experiments",
    response_model=list[ExperimentArtifact],
    responses={500: {"model": ErrorResponse}},
    tags=["experiments"],
)
def list_experiments(
    repository: ArtifactRepository = Depends(get_repository),
) -> list[ExperimentArtifact]:
    return repository.list_experiments()


@app.get(
    "/api/v1/experiments/{experiment_id}",
    response_model=ExperimentArtifact,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["experiments"],
)
def read_experiment(
    experiment_id: str,
    repository: ArtifactRepository = Depends(get_repository),
) -> ExperimentArtifact:
    return repository.get_experiment(experiment_id)


@app.get(
    "/api/v1/sample-efficiency",
    response_model=SampleEfficiencyArtifact,
    responses={500: {"model": ErrorResponse}},
    tags=["sample-efficiency"],
)
def read_sample_efficiency(
    repository: ArtifactRepository = Depends(get_repository),
) -> SampleEfficiencyArtifact:
    return repository.get_sample_efficiency()


@app.get(
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


# ---------------------------------------------------------------------------
# Exception handlers — единственное место перевода ошибок в HTTP-коды.
# ---------------------------------------------------------------------------
@app.exception_handler(ExperimentNotFoundError)
async def handle_experiment_not_found(
    request: Request, exc: ExperimentNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "experiment_not_found"},
    )


@app.exception_handler(UnsupportedChannelError)
async def handle_unsupported_channel(
    request: Request, exc: UnsupportedChannelError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "unsupported_channel"},
    )


@app.exception_handler(TimestampNotFoundError)
async def handle_timestamp_not_found(
    request: Request, exc: TimestampNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "timestamp_not_found"},
    )


@app.exception_handler(ArtifactNotFoundError)
async def handle_artifact_not_found(
    request: Request, exc: ArtifactNotFoundError
) -> JSONResponse:
    # Раздел 12 контракта: не раскрываем локальные пути на диске.
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Required artifact is missing",
            "error_type": "artifact_not_found",
        },
    )


@app.exception_handler(MalformedJSONError)
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


@app.exception_handler(InvalidArtifactError)
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
