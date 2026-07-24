from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .bitstream import CanonicalHuffmanCoder
from .normalization import NormalizationSpec
from .quantization import ScalarQuantizer
from .types import CodecConfig, CodecResult


class CodecHarness:
    """Deterministic local codec harness with exact symbol roundtrip checks."""

    def __init__(self, config: CodecConfig, normalization: NormalizationSpec) -> None:
        if config.channel_order != normalization.channel_order:
            raise ValueError("config channel order must match normalization channel order")
        self.config = config
        self.normalization = normalization
        self.quantizer = ScalarQuantizer(step=config.quantization_step)

    def encode_decode(self, tensor: np.ndarray, output_dir: Path) -> CodecResult:
        values = np.asarray(tensor, dtype=np.float32)
        self._validate_tensor(values)

        normalized = self._normalize(values)
        return self._encode_payload(
            input_tensor=values,
            latent=normalized,
            output_dir=output_dir,
        )

    def encode_latent(self, input_tensor: np.ndarray, latent: np.ndarray, output_dir: Path) -> CodecResult:
        values = np.asarray(input_tensor, dtype=np.float32)
        payload_values = np.asarray(latent, dtype=np.float32)
        self._validate_tensor(values)
        if payload_values.size == 0:
            raise ValueError("latent must not be empty")
        return self._encode_payload(
            input_tensor=values,
            latent=payload_values,
            output_dir=output_dir,
        )

    def _validate_tensor(self, values: np.ndarray) -> None:
        if values.ndim != 4:
            raise ValueError(f"tensor must have shape [batch, channel, height, width], got {values.shape}")
        expected_channels = len(self.config.channel_order)
        if values.shape[1] != expected_channels:
            raise ValueError(
                f"channel count mismatch: tensor has {values.shape[1]}, config expects {expected_channels}"
            )

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        mean = self.normalization.mean.reshape(1, -1, 1, 1)
        std = self.normalization.std.reshape(1, -1, 1, 1)
        return (values - mean) / std

    def _encode_payload(self, input_tensor: np.ndarray, latent: np.ndarray, output_dir: Path) -> CodecResult:
        symbols = self.quantizer.quantize(latent)
        coder = CanonicalHuffmanCoder.from_symbols(symbols.ravel())
        payload = coder.encode(symbols.ravel())
        decoded_symbols = coder.decode(payload, symbol_count=symbols.size).reshape(symbols.shape)

        if not np.array_equal(decoded_symbols, symbols):
            raise RuntimeError("quantized symbols do not match after encode/decode")

        decoded_latent = self.quantizer.dequantize(decoded_symbols).reshape(latent.shape)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bitstream_path = output_dir / "codec.bin"
        metadata_path = output_dir / "codec.json"
        bitstream_path.write_bytes(payload)

        tensor_ratio = float(input_tensor.size / max(latent.size, 1))
        serialized_ratio = float(input_tensor.nbytes / max(len(payload), 1))
        metadata = {
            "config": self.config.to_dict(),
            "normalization": self.normalization.to_dict(),
            "input": {
                "shape": list(input_tensor.shape),
                "dtype": str(input_tensor.dtype),
                "nbytes": int(input_tensor.nbytes),
            },
            "latent": {
                "shape": list(latent.shape),
                "dtype": str(latent.dtype),
                "nbytes": int(latent.nbytes),
                "value_count": int(latent.size),
            },
            "quantization": {
                "method": "uniform_scalar_round",
                "scale": float(self.quantizer.step),
                "symbol_dtype": str(symbols.dtype),
                "symbol_min": int(symbols.min()),
                "symbol_max": int(symbols.max()),
            },
            "entropy": {
                "coder": "canonical_huffman",
            },
            "compression": {
                "tensor_ratio": tensor_ratio,
                "latent_reduction_ratio": tensor_ratio,
                "serialized_ratio": serialized_ratio,
                "actual_compression_ratio": serialized_ratio,
                "symbol_count": int(symbols.size),
                "payload_bytes": len(payload),
                "bitstream_bytes": len(payload),
                "original_float32_bytes": int(input_tensor.nbytes),
                "quantized_latent_bytes": int(symbols.nbytes),
                "actual_bits_per_value": float((len(payload) * 8) / max(input_tensor.size, 1)),
            },
            "bitstream": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "path": bitstream_path.name,
            },
            "roundtrip": {
                "exact_symbol_match": True,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        return CodecResult(
            bitstream_path=bitstream_path,
            metadata_path=metadata_path,
            tensor_ratio=tensor_ratio,
            serialized_ratio=serialized_ratio,
            roundtrip_ok=True,
            metadata=metadata,
            decoded_latent=decoded_latent,
        )
