"""Strict, checkpoint-backed codec acceptance helpers.

This module intentionally sits below CLI and reporting code.  It does not fit
statistics, train a model, or access FastAPI.  A checkpoint must already carry
train-only normalization provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from era5_minimum.codec.bitstream import CanonicalHuffmanCoder
from era5_minimum.codec.normalization import (
    NormalizationSpec,
    denormalize_reconstruction,
    normalize_physical_tensor,
)
from era5_minimum.codec.quantization import ScalarQuantizer
from era5_minimum.data.channel_spec import CHANNEL_NAMES
from era5_minimum.models import ConvAutoencoder


_MAGIC = b"E5AC"
_FORMAT = "era5-minimum-acceptance-bitstream-v1"
_HEADER_PREFIX_BYTES = 8


class AcceptanceError(ValueError):
    """Raised when a checkpoint or codec contract cannot be accepted."""


@dataclass(frozen=True)
class SerializedLatent:
    """Decoded, exact quantized symbols and their self-describing payload."""

    symbols: np.ndarray
    latent_shape: tuple[int, int, int, int]
    payload: bytes
    header: dict[str, Any]
    header_bytes: int


@dataclass(frozen=True)
class CompressionResult:
    """One complete encode → bytes → decode pass in normalized value space."""

    reconstruction_normalized: np.ndarray
    symbols: np.ndarray
    bitstream: bytes
    header: dict[str, Any]
    encode_seconds: float
    decode_seconds: float
    latent_shape: tuple[int, int, int, int]


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 without loading a checkpoint into memory twice."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WeatherCodec:
    """Checkpoint-backed deterministic weather codec for acceptance only."""

    def __init__(
        self,
        *,
        model: ConvAutoencoder,
        normalization: NormalizationSpec,
        codec_config: dict[str, Any],
        checkpoint_sha256: str,
        device: torch.device,
        preprocessing: dict[str, Any],
    ) -> None:
        if tuple(normalization.channel_order) != tuple(CHANNEL_NAMES):
            raise AcceptanceError("checkpoint normalization channel order is not the official 28-channel contract")
        if model.in_channels != len(CHANNEL_NAMES):
            raise AcceptanceError(f"model in_channels={model.in_channels}, expected {len(CHANNEL_NAMES)}")
        self.model = model.eval()
        self.normalization = normalization
        self.codec_config = codec_config
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = device
        self.preprocessing = preprocessing
        self.quantizer = ScalarQuantizer(step=float(codec_config["quantization_step"]))

    @classmethod
    def load(cls, checkpoint_path: str | Path, *, device: str = "cpu") -> "WeatherCodec":
        """Load a compatible checkpoint strictly; partial state loads are rejected."""

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"state_dict", "model_config", "codec_config", "normalization", "channel_order"}
        missing = sorted(required - set(payload))
        if missing:
            raise AcceptanceError(f"checkpoint is missing required keys: {missing}")
        model_cfg = payload["model_config"]
        checkpoint_order = tuple(payload["channel_order"])
        if checkpoint_order != tuple(CHANNEL_NAMES):
            raise AcceptanceError("checkpoint channel_order does not match the official 28-channel contract")
        normalization = NormalizationSpec(**payload["normalization"])
        if not normalization.train_only:
            raise AcceptanceError("checkpoint normalization is not marked train-only")
        model = ConvAutoencoder(
            in_channels=int(model_cfg["in_channels"]),
            latent_channels=int(model_cfg["latent_channels"]),
        )
        incompatible = model.load_state_dict(payload["state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:  # pragma: no cover - strict=True raises first.
            raise AcceptanceError("checkpoint state_dict did not load strictly")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise AcceptanceError("CUDA was requested but is unavailable")
        model.to(resolved_device)
        return cls(
            model=model,
            normalization=normalization,
            codec_config=dict(payload["codec_config"]),
            checkpoint_sha256=sha256_file(path),
            device=resolved_device,
            preprocessing=dict(payload.get("preprocessing") or {}),
        )

    def preprocess(self, physical: np.ndarray, *, ocean_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Normalize one physical tensor without fitting or changing statistics."""

        return normalize_physical_tensor(
            physical,
            spec=self.normalization,
            ocean_mask=ocean_mask,
            sst_index=CHANNEL_NAMES.index("sst"),
        )

    def postprocess(self, reconstruction_normalized: np.ndarray, *, ocean_mask: np.ndarray) -> np.ndarray:
        """Denormalize and restore physical SST land NaNs."""

        physical = denormalize_reconstruction(
            reconstruction_normalized,
            spec=self.normalization,
            ocean_mask=None,
        )
        sst_index = CHANNEL_NAMES.index("sst")
        physical[:, sst_index] = np.where(
            np.asarray(ocean_mask, dtype=bool)[None, :, :],
            physical[:, sst_index],
            np.nan,
        )
        return physical

    def encode(self, model_input: np.ndarray) -> np.ndarray:
        """Encode exactly one normalized batch in inference mode."""

        tensor = torch.from_numpy(np.asarray(model_input, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            latent = self.model.encode(tensor)
        return latent.cpu().numpy().astype(np.float32, copy=False)

    def decode(self, latent: np.ndarray, *, output_shape: tuple[int, int]) -> np.ndarray:
        """Decode a dequantized latent without referring to an encoder object."""

        tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            reconstructed = self.model.decode(tensor, output_size=output_shape)
        return reconstructed.cpu().numpy().astype(np.float32, copy=False)

    def serialize(self, symbols: np.ndarray, *, original_shape: tuple[int, int, int, int]) -> bytes:
        """Serialize symbols plus all per-sample side information required for decode."""

        values = np.asarray(symbols, dtype=np.int32)
        if values.ndim != 4:
            raise AcceptanceError(f"quantized latent must be rank 4, got {values.shape}")
        entropy_payload = CanonicalHuffmanCoder.from_symbols(values.ravel()).encode(values.ravel())
        header = {
            "format": _FORMAT,
            "checkpoint_sha256": self.checkpoint_sha256,
            "codec_version": str(self.codec_config["version"]),
            "channel_order": list(CHANNEL_NAMES),
            "quantization": {"method": "uniform_scalar_round", "step": float(self.quantizer.step), "symbol_dtype": "int32"},
            "original_shape": list(original_shape),
            "latent_shape": list(values.shape),
            "entropy_payload_bytes": len(entropy_payload),
        }
        encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _MAGIC + struct.pack("<I", len(encoded_header)) + encoded_header + entropy_payload

    def deserialize(self, bitstream: bytes) -> SerializedLatent:
        """Recover exact symbols from a standalone per-sample byte stream."""

        if len(bitstream) < _HEADER_PREFIX_BYTES or bitstream[:4] != _MAGIC:
            raise AcceptanceError("invalid acceptance bitstream magic")
        header_length = struct.unpack("<I", bitstream[4:8])[0]
        header_end = _HEADER_PREFIX_BYTES + header_length
        if len(bitstream) < header_end:
            raise AcceptanceError("acceptance bitstream header is truncated")
        try:
            header = json.loads(bitstream[_HEADER_PREFIX_BYTES:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcceptanceError("acceptance bitstream header is malformed") from exc
        if header.get("format") != _FORMAT:
            raise AcceptanceError(f"unsupported bitstream format: {header.get('format')!r}")
        if header.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise AcceptanceError("bitstream requires a different checkpoint")
        if tuple(header.get("channel_order", ())) != tuple(CHANNEL_NAMES):
            raise AcceptanceError("bitstream channel order does not match this codec")
        if float(header.get("quantization", {}).get("step", -1.0)) != float(self.quantizer.step):
            raise AcceptanceError("bitstream quantization step does not match checkpoint")
        latent_shape = tuple(int(value) for value in header.get("latent_shape", ()))
        if len(latent_shape) != 4 or latent_shape[1] != self.model.latent_channels:
            raise AcceptanceError("bitstream latent shape is incompatible with checkpoint")
        payload = bitstream[header_end:]
        if len(payload) != int(header.get("entropy_payload_bytes", -1)):
            raise AcceptanceError("bitstream entropy payload length does not match header")
        coder = CanonicalHuffmanCoder.from_symbols(np.array([0], dtype=np.int32))
        symbols = coder.decode(payload, symbol_count=math.prod(latent_shape)).reshape(latent_shape)
        return SerializedLatent(
            symbols=symbols,
            latent_shape=latent_shape,
            payload=payload,
            header=header,
            header_bytes=header_end,
        )

    def compress(self, model_input: np.ndarray) -> tuple[bytes, np.ndarray, float, tuple[int, int, int, int]]:
        """Encode, quantize and serialize a normalized tensor."""

        started = time.perf_counter()
        latent = self.encode(model_input)
        symbols = self.quantizer.quantize(latent)
        original_shape = tuple(int(value) for value in model_input.shape)
        return self.serialize(symbols, original_shape=original_shape), symbols, time.perf_counter() - started, tuple(int(value) for value in latent.shape)

    def decompress(self, bitstream: bytes) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        """Deserialize and decode using only this checkpoint-backed codec context."""

        started = time.perf_counter()
        decoded = self.deserialize(bitstream)
        latent = self.quantizer.dequantize(decoded.symbols).reshape(decoded.latent_shape)
        output_shape = tuple(int(value) for value in decoded.header["original_shape"][-2:])
        reconstruction = self.decode(latent, output_shape=output_shape)
        return reconstruction, decoded.symbols, time.perf_counter() - started, decoded.header

    def reconstruct(self, model_input: np.ndarray) -> CompressionResult:
        """Run the complete codec path and enforce exact symbol roundtrip."""

        bitstream, encoded_symbols, encode_seconds, latent_shape = self.compress(model_input)
        reconstruction, decoded_symbols, decode_seconds, header = self.decompress(bitstream)
        if not np.array_equal(encoded_symbols, decoded_symbols):
            raise AcceptanceError("quantized symbols changed after serialization")
        return CompressionResult(
            reconstruction_normalized=reconstruction,
            symbols=decoded_symbols,
            bitstream=bitstream,
            header=header,
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
            latent_shape=latent_shape,
        )


def compression_metrics(*, original_shape: tuple[int, int, int, int], bitstream: bytes, header: dict[str, Any]) -> dict[str, Any]:
    """Return only actual serialized compression accounting, never tensor ratios."""

    input_bits = 32 * math.prod(original_shape)
    total_bytes = len(bitstream)
    header_bytes = _HEADER_PREFIX_BYTES + len(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    payload_bytes = total_bytes - header_bytes
    ratio = input_bits / (8 * total_bytes) if total_bytes else None
    return {
        "input_bits_float32": input_bits,
        "payload_bytes": payload_bytes,
        "header_bytes": header_bytes,
        "side_information_bytes": header_bytes,
        "total_bitstream_bytes": total_bytes,
        "bits_per_value": (8 * total_bytes) / math.prod(original_shape),
        "compression_ratio": ratio,
        "target_32x_reached": bool(ratio is not None and ratio >= 32.0),
        "target_64x_reached": bool(ratio is not None and ratio >= 64.0),
    }


def evaluate_physical_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    latitudes: np.ndarray,
    ocean_mask: np.ndarray,
    train_std: np.ndarray | None,
) -> dict[str, Any]:
    """Per-channel latitude-weighted physical metrics with an SST ocean mask."""

    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if original.shape != reconstructed.shape or original.ndim != 4 or original.shape[1] != len(CHANNEL_NAMES):
        raise AcceptanceError("physical metric tensors must have matching [batch, 28, height, width] shape")
    weights = np.clip(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0, None).reshape(1, 1, -1, 1)
    ocean = np.asarray(ocean_mask, dtype=bool)
    if ocean.shape != original.shape[-2:]:
        raise AcceptanceError("ocean mask shape does not match metric tensors")
    std = None if train_std is None else np.asarray(train_std, dtype=np.float64)
    if std is not None and std.shape != (len(CHANNEL_NAMES),):
        raise AcceptanceError("train_std must contain 28 values")

    per_channel: dict[str, Any] = {}
    for index, name in enumerate(CHANNEL_NAMES):
        source = original[:, index : index + 1]
        prediction = reconstructed[:, index : index + 1]
        valid = np.isfinite(source) & np.isfinite(prediction)
        if name == "sst":
            valid &= ocean[None, None, :, :]
        error = prediction - source
        valid_weights = weights * valid
        total_weight = float(valid_weights.sum())
        if total_weight == 0.0:
            weighted_rmse = weighted_mae = bias = None
        else:
            weighted_rmse = float(np.sqrt(np.sum(error**2 * valid_weights) / total_weight))
            weighted_mae = float(np.sum(np.abs(error) * valid_weights) / total_weight)
            bias = float(np.sum(error * valid_weights) / total_weight)
        original_valid = source[np.isfinite(source)]
        reconstructed_valid = prediction[np.isfinite(prediction)]
        peak = float(np.max(original_valid) - np.min(original_valid)) if original_valid.size else None
        psnr = None
        if weighted_rmse is not None and weighted_rmse > 0 and peak is not None and peak > 0:
            psnr = float(20 * np.log10(peak / weighted_rmse))
        per_channel[name] = {
            "latitude_weighted_rmse": weighted_rmse,
            "latitude_weighted_mae": weighted_mae,
            "bias": bias,
            "psnr": psnr,
            "psnr_peak_convention": "per-sample physical valid original range (max-min)",
            "valid_point_count": int(valid.sum()),
            "original_nan_count": int(np.isnan(source).sum()),
            "reconstructed_nan_count": int(np.isnan(prediction).sum()),
            "original_infinity_count": int(np.isinf(source).sum()),
            "reconstructed_infinity_count": int(np.isinf(prediction).sum()),
            "original_min": float(np.min(original_valid)) if original_valid.size else None,
            "original_max": float(np.max(original_valid)) if original_valid.size else None,
            "reconstructed_min": float(np.min(reconstructed_valid)) if reconstructed_valid.size else None,
            "reconstructed_max": float(np.max(reconstructed_valid)) if reconstructed_valid.size else None,
            "nrmse": None if std is None or weighted_rmse is None else float(weighted_rmse / std[index]),
            "nrmse_status": "not_available_without_train_std" if std is None else "available",
        }
    return {
        "per_channel": per_channel,
        "aggregates": {
            "surface_score": None,
            "pressure_score": None,
            "overall_score": None,
            "status": "not_available_without_a_dimensionless_predeclared_cross_variable_score",
        },
    }
