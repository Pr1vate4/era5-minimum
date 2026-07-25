"""HTTP contract tests for the checkpoint-backed interactive codec API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from era5_minimum.api.app import app, get_codec_service
from era5_minimum.api.codec_service import CodecArtifact, CompletedCodecJob


class _CodecServiceStub:
    """Small deterministic substitute; model inference belongs to acceptance tests."""

    def __init__(self) -> None:
        self.job: CompletedCodecJob | None = None
        self.received: tuple[bytes, str, int] | None = None

    def status(self) -> dict:
        return {
            "ready": True,
            "checkpoint": "fixture-sha256",
            "model_name": "fixture N128-equivalent",
            "message": "ready",
            "supported_target_ratios": [32],
        }

    def create_job(self, *, payload: bytes, filename: str, target_ratio: int) -> dict:
        self.received = (payload, filename, target_ratio)
        identifier = "codec-fixture"
        root = f"/api/v1/codec/jobs/{identifier}"
        response = {
            "id": identifier,
            "status": "completed",
            "progress": 1.0,
            "message": "done",
            "error": None,
            "metrics": {
                "serialized_compression_ratio": 128.0,
                "tensor_compression_ratio": 32.0,
                "bitstream_bytes": 42,
                "exact_roundtrip": True,
                "encode_seconds": 0.1,
                "decode_seconds": 0.2,
            },
            "downloads": {"bitstream": f"{root}/bitstream", "reconstruction": f"{root}/reconstruction"},
            "previews": {"original": f"{root}/preview/original.png", "reconstruction": f"{root}/preview/reconstruction.png"},
        }
        self.job = CompletedCodecJob(
            identifier=identifier,
            payload=response,
            artifact=CodecArtifact(b"stream", b"npz", b"original", b"reconstruction"),
            created_at=0.0,
        )
        return response

    def get_job(self, identifier: str) -> CompletedCodecJob | None:
        return self.job if self.job and identifier == self.job.identifier else None


def test_interactive_codec_routes_match_frontend_contract() -> None:
    service = _CodecServiceStub()
    app.dependency_overrides[get_codec_service] = lambda: service
    try:
        client = TestClient(app)
        assert client.get("/api/v1/codec/status").json()["ready"] is True

        created = client.post(
            "/api/v1/codec/jobs",
            files={"file": ("frame.npz", b"fixture-npz", "application/octet-stream")},
            data={"target_ratio": "32"},
        )
        assert created.status_code == 201
        payload = created.json()
        assert payload["metrics"]["serialized_compression_ratio"] == 128.0
        assert service.received == (b"fixture-npz", "frame.npz", 32)

        assert client.get(payload["downloads"]["bitstream"]).content == b"stream"
        assert client.get(payload["downloads"]["reconstruction"]).content == b"npz"
        assert client.get(payload["previews"]["original"]).content == b"original"
        assert client.get(payload["previews"]["reconstruction"]).content == b"reconstruction"
        assert client.get("/api/v1/codec/jobs/missing").status_code == 404
    finally:
        app.dependency_overrides.clear()
