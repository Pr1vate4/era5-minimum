from __future__ import annotations

import numpy as np

from era5_minimum.codec.normalization import NormalizationSpec
from era5_minimum.codec.quantization import ScalarQuantizer


def test_scalar_quantizer_roundtrips_integer_symbols() -> None:
    quantizer = ScalarQuantizer(step=0.25, zero_point=0)
    values = np.array([[-0.25, 0.0, 0.25]], dtype=np.float32)

    symbols = quantizer.quantize(values)

    assert np.array_equal(symbols, np.array([[-1, 0, 1]], dtype=np.int32))
    assert np.array_equal(quantizer.dequantize(symbols), values)


def test_normalization_spec_keeps_train_only_provenance() -> None:
    spec = NormalizationSpec(
        channel_order=("t2m", "msl"),
        mean=np.array([1.0, 2.0], dtype=np.float32),
        std=np.array([3.0, 4.0], dtype=np.float32),
        source_manifest_sha256="abc123",
        train_only=True,
    )

    payload = spec.to_dict()

    assert payload["train_only"] is True
    assert payload["source_manifest_sha256"] == "abc123"
