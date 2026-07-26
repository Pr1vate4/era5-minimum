"""FastAPI routes for timestamp-driven real ERA5 frame preparation."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from era5_minimum.api.codec_service import (
    CodecInputError,
    CodecService,
    CodecUnavailableError,
    get_codec_service,
)
from era5_minimum.api.era5_frame_schemas import (
    CodecJobResponse,
    Era5ChannelMetadata,
    Era5ErrorResponse,
    Era5FrameMetadata,
    Era5FrameValidationMetadata,
    Era5LatitudeMetadata,
    Era5LongitudeMetadata,
    Era5TimestampList,
)
from era5_minimum.api.era5_frame_service import (
    DatasetUnavailable,
    Era5FrameError,
    Era5FrameService,
    PreparedEra5Frame,
    serialize_frame_npz,
)


router = APIRouter(prefix="/api/v1/era5", tags=["era5-frames"])


@lru_cache(maxsize=1)
def get_era5_frame_service() -> Era5FrameService:
    """Open the process-local configured Zarr frame source."""

    provider = os.getenv("ERA5_WEATHER_PROVIDER", "zarr")
    if provider != "zarr":
        raise DatasetUnavailable(
            f"ERA5 frame preparation requires the zarr provider, got {provider!r}"
        )
    return Era5FrameService.from_root(
        os.getenv("ERA5_DATASET_ROOT", "data/era5_28ch_demo"),
        os.getenv("ERA5_DATASET_SPLIT", "validation"),
    )


def _metadata(frame: PreparedEra5Frame) -> Era5FrameMetadata:
    return Era5FrameMetadata(
        timestamp=frame.timestamp,
        shape=tuple(int(value) for value in frame.tensor.shape),
        dtype="float32",
        dataset_id=frame.dataset_id,
        dataset_source=frame.dataset_source,
        dataset_split=frame.dataset_split,
        schema_version=frame.schema_version,
        channels=[
            Era5ChannelMetadata(
                index=channel.index,
                name=channel.name,
                source_variable=channel.source_name,
                level=channel.level,
                unit=channel.units,
            )
            for channel in frame.channels
        ],
        latitude=Era5LatitudeMetadata(
            size=int(frame.latitude.size),
            order="south_to_north",
            minimum=float(frame.latitude[0]),
            maximum=float(frame.latitude[-1]),
        ),
        longitude=Era5LongitudeMetadata(
            size=int(frame.longitude.size),
            order="west_to_east",
            range=(float(frame.longitude[0]), float(frame.longitude[-1])),
            periodic=True,
        ),
        validation=Era5FrameValidationMetadata(
            valid=frame.validation.valid,
            missing_values=frame.validation.missing_values,
            sst_masked_values=frame.validation.sst_masked_values,
            sst_ocean_missing_values=(
                frame.validation.sst_ocean_missing_values
            ),
            invalid_values=frame.validation.invalid_values,
        ),
    )


ERROR_RESPONSES = {
    404: {"model": Era5ErrorResponse},
    422: {"model": Era5ErrorResponse},
    503: {"model": Era5ErrorResponse},
}


@router.get(
    "/timestamps",
    response_model=Era5TimestampList,
    responses=ERROR_RESPONSES,
    summary="List available ERA5 timestamps",
    description="Returns exact timestamps from the configured local Zarr split.",
)
def list_era5_timestamps(
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
    service: Era5FrameService = Depends(get_era5_frame_service),
) -> Era5TimestampList:
    timestamps, total = service.list_timestamps(
        start=start,
        end=end,
        offset=offset,
        limit=limit,
    )
    return Era5TimestampList(
        timestamps=list(timestamps),
        count=len(timestamps),
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/frames/{timestamp}",
    response_model=Era5FrameMetadata,
    responses=ERROR_RESPONSES,
    summary="Validate one ERA5 model frame",
    description=(
        "Loads one exact timestamp, prepares the canonical physical "
        "float32 [1, 28, 360, 720] tensor, and returns metadata only."
    ),
)
def get_era5_frame_metadata(
    timestamp: datetime,
    service: Era5FrameService = Depends(get_era5_frame_service),
) -> Era5FrameMetadata:
    return _metadata(service.prepare_frame(timestamp))


@router.get(
    "/frames/{timestamp}/npz",
    responses={
        200: {"content": {"application/x-npz": {}}},
        **ERROR_RESPONSES,
    },
    summary="Download one canonical ERA5 NPZ frame",
    description="Exports the same data/channel_order keys accepted by the codec upload API.",
)
def download_era5_frame_npz(
    timestamp: datetime,
    service: Era5FrameService = Depends(get_era5_frame_service),
) -> Response:
    frame = service.prepare_frame(timestamp)
    safe_timestamp = frame.timestamp.strftime("%Y%m%dT%H%M%SZ")
    return Response(
        serialize_frame_npz(frame),
        media_type="application/x-npz",
        headers={
            "Content-Disposition": (
                f'attachment; filename="era5-{safe_timestamp}.npz"'
            ),
            "X-ERA5-Dataset": frame.dataset_id,
        },
    )


@router.post(
    "/frames/{timestamp}/compress",
    status_code=201,
    response_model=CodecJobResponse,
    responses=ERROR_RESPONSES,
    summary="Compress one real ERA5 timestamp",
    description=(
        "Prepares an exact Zarr timestamp and invokes the existing in-process "
        "codec service without an intermediate file or self-HTTP request."
    ),
)
def compress_era5_frame(
    timestamp: datetime,
    frame_service: Era5FrameService = Depends(get_era5_frame_service),
    codec_service: CodecService = Depends(get_codec_service),
) -> dict:
    frame = frame_service.prepare_frame(timestamp)
    try:
        return codec_service.create_job_from_frame(
            values=frame.tensor,
            channel_order=tuple(channel.name for channel in frame.channels),
            ocean_mask=frame.ocean_mask,
            target_ratio=32,
            source={
                "type": "era5",
                "timestamp": frame.timestamp.isoformat().replace("+00:00", "Z"),
                "dataset_id": frame.dataset_id,
            },
        )
    except CodecUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CodecInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def handle_era5_frame_error(
    request: Request,
    exc: Era5FrameError,
) -> JSONResponse:
    """Return stable domain errors without exposing Python tracebacks."""

    return JSONResponse(status_code=exc.status_code, content=exc.payload())
