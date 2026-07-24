from __future__ import annotations

import numpy as np
import pytest

from era5_minimum.codec.bitstream import BitstreamError, CanonicalHuffmanCoder


def test_canonical_huffman_roundtrips_exact_symbols() -> None:
    symbols = np.array([3, 3, 3, 1, 1, 2, 2, 2], dtype=np.int32)
    coder = CanonicalHuffmanCoder.from_symbols(symbols)

    payload = coder.encode(symbols)
    decoded = coder.decode(payload, symbol_count=symbols.size)

    assert np.array_equal(decoded, symbols)


def test_canonical_huffman_rejects_truncated_payload() -> None:
    symbols = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    coder = CanonicalHuffmanCoder.from_symbols(symbols)
    payload = coder.encode(symbols)

    with pytest.raises(BitstreamError):
        coder.decode(payload[:-1], symbol_count=symbols.size)


def test_canonical_huffman_roundtrips_single_symbol_stream() -> None:
    symbols = np.array([7, 7, 7, 7], dtype=np.int32)
    coder = CanonicalHuffmanCoder.from_symbols(symbols)

    payload = coder.encode(symbols)
    decoded = coder.decode(payload, symbol_count=symbols.size)

    assert np.array_equal(decoded, symbols)
