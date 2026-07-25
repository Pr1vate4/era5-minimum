"""In-memory serving adapter for the verified N128-equivalent ERA5 codec.

The service deliberately accepts a single, canonical physical ERA5 frame.  It
does not fit statistics or silently reorder channels: both operations would
make an interactive result incomparable with the accepted model evaluation.
"""

from __future__ import annotations

import io
import os
import struct
import time
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from era5_minimum.api.monitoring import record_codec_job
from era5_minimum.codec.acceptance import AcceptanceError, WeatherCodec, compression_metrics
from era5_minimum.data.channel_spec import CHANNEL_NAMES


CANONICAL_FRAME_SHAPE = (1, len(CHANNEL_NAMES), 360, 720)
MAX_UPLOAD_BYTES = int(os.getenv("ERA5_CODEC_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))
DEFAULT_MODEL_DIR = Path(os.getenv("ERA5_CODEC_MODEL_DIR", "artifacts/model-n32"))
_MAX_JOBS = 4


class CodecInputError(ValueError):
    """Raised for an upload that is not a canonical physical ERA5 frame."""


class CodecUnavailableError(RuntimeError):
    """Raised when the installed checkpoint cannot serve requests."""


@dataclass(frozen=True)
class CodecArtifact:
    """Binary outputs retained for a short in-memory download window."""

    bitstream: bytes
    reconstruction: bytes
    original_preview: bytes
    reconstruction_preview: bytes


@dataclass(frozen=True)
class CompletedCodecJob:
    """A completed synchronous browser request in the frontend job contract."""

    identifier: str
    payload: dict[str, Any]
    artifact: CodecArtifact
    created_at: float


def _png_gray(values: np.ndarray) -> bytes:
    """Render a deterministic greyscale PNG without a plotting dependency."""

    field = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(field)
    if not finite.any():
        pixels = np.zeros(field.shape, dtype=np.uint8)
    else:
        low, high = np.quantile(field[finite], (0.02, 0.98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(field)), float(np.nanmax(field))
        scale = 1.0 if high <= low else 255.0 / (high - low)
        pixels = np.where(finite, np.clip((field - low) * scale, 0, 255), 0).astype(np.uint8)

    height, width = pixels.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, level=6)) + chunk(b"IEND", b"")


def _read_npz(payload: bytes) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Parse a bounded NPZ without allowing object deserialization."""

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "data" not in archive:
                raise CodecInputError("NPZ must contain a float32 'data' array")
            values = np.asarray(archive["data"])
            order = None
            if "channel_order" in archive:
                order = tuple(str(value) for value in archive["channel_order"].tolist())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise CodecInputError("cannot read a safe NPZ archive") from exc

    if values.shape == CANONICAL_FRAME_SHAPE[1:]:
        values = values[None, ...]
    if values.shape != CANONICAL_FRAME_SHAPE:
        raise CodecInputError(
            "data must have shape [1, 28, 360, 720] (or [28, 360, 720]), "
            f"got {tuple(values.shape)}"
        )
    if values.dtype != np.float32:
        raise CodecInputError(f"data must use float32 physical units, got {values.dtype}")
    if order is not None and order != CHANNEL_NAMES:
        raise CodecInputError("channel_order does not match the canonical 28-channel ERA5 contract")
    return values, order


class CodecService:
    """Load one immutable checkpoint and expose bounded browser jobs."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._codec: WeatherCodec | None = None
        self._load_error: str | None = None
        self._jobs: dict[str, CompletedCodecJob] = {}
        self._lock = Lock()

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "model.ckpt"

    def _get_codec(self) -> WeatherCodec:
        with self._lock:
            if self._codec is not None:
                return self._codec
            if self._load_error is not None:
                raise CodecUnavailableError(self._load_error)
            try:
                self._codec = WeatherCodec.load(self.checkpoint_path)
            except (AcceptanceError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                self._load_error = f"N128-equivalent checkpoint is unavailable: {exc}"
                raise CodecUnavailableError(self._load_error) from exc
            return self._codec

    def status(self) -> dict[str, Any]:
        """Return readiness without failing the page when a model is absent."""

        try:
            codec = self._get_codec()
        except CodecUnavailableError as exc:
            return {"ready": False, "checkpoint": None, "model_name": None, "message": str(exc), "supported_target_ratios": [32]}
        return {
            "ready": True,
            "checkpoint": codec.checkpoint_sha256,
            "model_name": "ConvAE N128-equivalent (28 channels, 0.5°; 32 фактических кадров)",
            "message": "N128-equivalent checkpoint loaded; serialized rate is measured per uploaded frame.",
            "supported_target_ratios": [32],
        }

    def create_job(self, *, payload: bytes, filename: str, target_ratio: int) -> dict[str, Any]:
        """Compress one uploaded frame and retain its downloads in process memory."""

        if target_ratio != 32:
            raise CodecInputError("only the installed N128-equivalent profile is available; 64× is not a supported tensor profile")
        if not filename.lower().endswith(".npz"):
            raise CodecInputError("only canonical .npz uploads are supported")
        if not payload:
            raise CodecInputError("uploaded file is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise CodecInputError(f"uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit")

        values, _ = _read_npz(payload)
        codec = self._get_codec()
        started = time.perf_counter()
        try:
            ocean_mask = np.isfinite(values[0, CHANNEL_NAMES.index("sst")])
            model_input, _, _ = codec.preprocess(values, ocean_mask=ocean_mask)
            result = codec.reconstruct(model_input)
            physical = codec.postprocess(result.reconstruction_normalized, ocean_mask=ocean_mask)
        except (AcceptanceError, ValueError, RuntimeError) as exc:
            record_codec_job(status="error")
            raise CodecInputError(f"model could not process the frame: {exc}") from exc

        metrics = compression_metrics(
            original_shape=tuple(int(value) for value in values.shape),
            bitstream=result.bitstream,
            header=result.header,
        )
        reconstruction_buffer = io.BytesIO()
        np.savez_compressed(
            reconstruction_buffer,
            data=physical,
            channel_order=np.asarray(CHANNEL_NAMES),
            source_checkpoint_sha256=codec.checkpoint_sha256,
        )
        identifier = uuid.uuid4().hex
        root = f"/api/v1/codec/jobs/{identifier}"
        job_payload = {
            "id": identifier,
            "status": "completed",
            "progress": 1.0,
            "message": "Frame compressed and reconstructed by the N128-equivalent checkpoint.",
            "error": None,
            "metrics": {
                "serialized_compression_ratio": metrics["compression_ratio"],
                "tensor_compression_ratio": float(np.prod(values.shape) / np.prod(result.latent_shape)),
                "bitstream_bytes": len(result.bitstream),
                "exact_roundtrip": True,
                "encode_seconds": result.encode_seconds,
                "decode_seconds": result.decode_seconds,
            },
            "downloads": {"bitstream": f"{root}/bitstream", "reconstruction": f"{root}/reconstruction"},
            "previews": {"original": f"{root}/preview/original.png", "reconstruction": f"{root}/preview/reconstruction.png"},
        }
        artifact = CodecArtifact(
            bitstream=result.bitstream,
            reconstruction=reconstruction_buffer.getvalue(),
            original_preview=_png_gray(values[0, 0]),
            reconstruction_preview=_png_gray(physical[0, 0]),
        )
        with self._lock:
            self._jobs[identifier] = CompletedCodecJob(identifier, job_payload, artifact, time.time())
            while len(self._jobs) > _MAX_JOBS:
                oldest = min(self._jobs.values(), key=lambda job: job.created_at)
                del self._jobs[oldest.identifier]
        record_codec_job(status="success", elapsed_seconds=time.perf_counter() - started)
        return job_payload

    def get_job(self, identifier: str) -> CompletedCodecJob | None:
        """Return a job while its bounded in-memory download window is open."""

        with self._lock:
            return self._jobs.get(identifier)
