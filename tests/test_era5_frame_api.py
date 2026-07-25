"""Integration contract tests for timestamp-driven ERA5 API routes."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from era5_minimum.api.codec_service import get_codec_service
from era5_minimum.api.era5_frame_service import (
    Era5FrameError,
    FrameValidationSummary,
    PreparedEra5Frame,
)
from era5_minimum.api.era5_frames import (
    get_era5_frame_service,
    handle_era5_frame_error,
    router,
)
from era5_minimum.data.channel_spec import CHANNEL_NAMES, CHANNEL_SPEC
from era5_minimum.data.grids import get_target_grid_05
from era5_minimum.data.model_input import CANONICAL_FRAME_SHAPE


TIMESTAMP = datetime(2020, 1, 1, 18, tzinfo=UTC)


class _FrameServiceStub:
    def __init__(self) -> None:
        target = get_target_grid_05()
        tensor = np.zeros(CANONICAL_FRAME_SHAPE, dtype=np.float32)
        mask = np.ones(CANONICAL_FRAME_SHAPE[-2:], dtype=bool)
        self.frame = PreparedEra5Frame(
            timestamp=TIMESTAMP,
            tensor=tensor,
            ocean_mask=mask,
            latitude=target.latitude.values,
            longitude=target.longitude.values,
            channels=CHANNEL_SPEC,
            dataset_id="real-fixture",
            dataset_source="fixture://weatherbench2",
            dataset_split="validation",
            schema_version="weatherbench2-28ch-0p5-v1",
            validation=FrameValidationSummary(True, 0, 0, 0, 0),
        )

    def list_timestamps(self, *, start, end, offset: int, limit: int):
        values = (TIMESTAMP,)
        return values[offset : offset + limit], len(values)

    def prepare_frame(self, timestamp: datetime) -> PreparedEra5Frame:
        assert timestamp == TIMESTAMP
        return self.frame


class _CodecServiceStub:
    def __init__(self) -> None:
        self.received = None

    def create_job_from_frame(
        self,
        *,
        values,
        channel_order,
        ocean_mask,
        target_ratio,
        source,
    ):
        self.received = (values, channel_order, ocean_mask, target_ratio, source)
        return {
            "id": "timestamp-job",
            "status": "completed",
            "progress": 1.0,
            "message": "done",
            "error": None,
            "metrics": {
                "serialized_compression_ratio": 64.0,
                "tensor_compression_ratio": 32.0,
                "bitstream_bytes": 1024,
                "exact_roundtrip": True,
                "encode_seconds": 0.1,
                "decode_seconds": 0.1,
            },
            "downloads": {
                "bitstream": "/api/v1/codec/jobs/timestamp-job/bitstream",
                "reconstruction": (
                    "/api/v1/codec/jobs/timestamp-job/reconstruction"
                ),
            },
            "previews": {
                "original": (
                    "/api/v1/codec/jobs/timestamp-job/preview/original.png"
                ),
                "reconstruction": (
                    "/api/v1/codec/jobs/timestamp-job/preview/reconstruction.png"
                ),
            },
            "source": {
                "type": "era5",
                "timestamp": "2020-01-01T18:00:00Z",
                "dataset_id": "real-fixture",
            },
        }


def test_timestamp_metadata_npz_and_compress_routes_share_services() -> None:
    frame_service = _FrameServiceStub()
    codec_service = _CodecServiceStub()
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.add_exception_handler(Era5FrameError, handle_era5_frame_error)
    test_app.dependency_overrides[get_era5_frame_service] = lambda: frame_service
    test_app.dependency_overrides[get_codec_service] = lambda: codec_service
    try:
        client = TestClient(test_app)
        timestamps = client.get("/api/v1/era5/timestamps")
        assert timestamps.status_code == 200
        assert timestamps.json()["timestamps"] == ["2020-01-01T18:00:00Z"]

        path = "/api/v1/era5/frames/2020-01-01T18:00:00Z"
        metadata = client.get(path)
        assert metadata.status_code == 200
        payload = metadata.json()
        assert payload["shape"] == [1, 28, 360, 720]
        assert payload["dtype"] == "float32"
        assert [channel["name"] for channel in payload["channels"]] == list(
            CHANNEL_NAMES
        )

        exported = client.get(f"{path}/npz")
        assert exported.status_code == 200
        with np.load(io.BytesIO(exported.content), allow_pickle=False) as archive:
            assert set(archive.files) == {"data", "channel_order"}
            assert archive["data"].shape == (1, 28, 360, 720)
            assert archive["data"].dtype == np.float32
            assert tuple(archive["channel_order"].tolist()) == CHANNEL_NAMES

        compressed = client.post(f"{path}/compress")
        assert compressed.status_code == 201
        assert compressed.json()["id"] == "timestamp-job"
        assert codec_service.received is not None
        values, order, mask, ratio, source = codec_service.received
        assert values is frame_service.frame.tensor
        assert order == CHANNEL_NAMES
        assert mask is frame_service.frame.ocean_mask
        assert ratio == 32
        assert source == {
            "type": "era5",
            "timestamp": "2020-01-01T18:00:00Z",
            "dataset_id": "real-fixture",
        }
    finally:
        test_app.dependency_overrides.clear()
