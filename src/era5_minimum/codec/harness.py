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
        symbols = self.quantizer.quantize(normalized)
        coder = CanonicalHuffmanCoder.from_symbols(symbols.ravel())
        payload = coder.encode(symbols.ravel())
        decoded_symbols = coder.decode(payload, symbol_count=symbols.size).reshape(symbols.shape)

        if not np.array_equal(decoded_symbols, symbols):
            raise RuntimeError("quantized symbols do not match after encode/decode")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bitstream_path = output_dir / "codec.bin"
        metadata_path = output_dir / "codec.json"
        bitstream_path.write_bytes(payload)

        tensor_ratio = float(values.size / max(symbols.size, 1))
        serialized_ratio = float(values.nbytes / max(len(payload), 1))
        metadata = {
            "config": self.config.to_dict(),
            "normalization": self.normalization.to_dict(),
            "input": {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "nbytes": int(values.nbytes),
            },
            "compression": {
                "tensor_ratio": tensor_ratio,
                "serialized_ratio": serialized_ratio,
                "symbol_count": int(symbols.size),
                "payload_bytes": len(payload),
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
