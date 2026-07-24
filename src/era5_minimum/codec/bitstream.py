from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .huffman import build_canonical_codes, build_code_lengths

_HEADER_SIZE = 4
_FORMAT_VERSION = "era5-minimum-canonical-huffman-v1"


class BitstreamError(ValueError):
    """Raised when an encoded payload cannot be decoded safely."""


class _BitWriter:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current = 0
        self._used_bits = 0

    def write(self, code: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            bit = (code >> shift) & 1
            self._current = (self._current << 1) | bit
            self._used_bits += 1
            if self._used_bits == 8:
                self._buffer.append(self._current)
                self._current = 0
                self._used_bits = 0

    def finish(self) -> bytes:
        if self._used_bits:
            self._current <<= 8 - self._used_bits
            self._buffer.append(self._current)
            self._current = 0
            self._used_bits = 0
        return bytes(self._buffer)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._byte_index = 0
        self._bit_index = 0

    def read_bit(self) -> int:
        if self._byte_index >= len(self._payload):
            raise BitstreamError("bitstream payload is truncated")
        current = self._payload[self._byte_index]
        bit = (current >> (7 - self._bit_index)) & 1
        self._bit_index += 1
        if self._bit_index == 8:
            self._byte_index += 1
            self._bit_index = 0
        return bit


@dataclass(frozen=True)
class CanonicalHuffmanCoder:
    """Canonical Huffman encoder/decoder with self-describing header."""

    code_lengths: dict[int, int]
    codes: dict[int, tuple[int, int]]

    @classmethod
    def from_symbols(cls, symbols: np.ndarray) -> "CanonicalHuffmanCoder":
        flat = np.asarray(symbols, dtype=np.int32).ravel()
        counts = Counter(int(value) for value in flat)
        code_lengths = build_code_lengths(counts)
        codes = build_canonical_codes(code_lengths)
        return cls(code_lengths=code_lengths, codes=codes)

    def encode(self, symbols: np.ndarray) -> bytes:
        flat = np.asarray(symbols, dtype=np.int32).ravel()
        writer = _BitWriter()
        for value in flat:
            code, width = self.codes[int(value)]
            writer.write(code, width)
        header = json.dumps(
            {
                "format": _FORMAT_VERSION,
                "symbol_count": int(flat.size),
                "code_lengths": {str(symbol): width for symbol, width in self.code_lengths.items()},
            },
            sort_keys=True,
        ).encode("utf-8")
        return struct.pack("<I", len(header)) + header + writer.finish()

    def decode(self, payload: bytes, symbol_count: int) -> np.ndarray:
        if len(payload) < _HEADER_SIZE:
            raise BitstreamError("bitstream is truncated")

        header_length = struct.unpack("<I", payload[:_HEADER_SIZE])[0]
        if len(payload) < _HEADER_SIZE + header_length:
            raise BitstreamError("bitstream header is truncated")

        header_bytes = payload[_HEADER_SIZE : _HEADER_SIZE + header_length]
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise BitstreamError("bitstream header is malformed") from error

        if header.get("format") != _FORMAT_VERSION:
            raise BitstreamError(f"unsupported bitstream format: {header.get('format')!r}")
        if int(header.get("symbol_count", -1)) != int(symbol_count):
            raise BitstreamError("symbol count does not match bitstream header")

        try:
            code_lengths = {int(symbol): int(width) for symbol, width in header["code_lengths"].items()}
        except (KeyError, TypeError, ValueError) as error:
            raise BitstreamError("bitstream code lengths are malformed") from error

        decode_table = {
            (width, code): symbol for symbol, (code, width) in build_canonical_codes(code_lengths).items()
        }
        max_width = max(width for width, _code in decode_table)
        reader = _BitReader(payload[_HEADER_SIZE + header_length :])
        symbols: list[int] = []

        for _ in range(symbol_count):
            code = 0
            matched = False
            for width in range(1, max_width + 1):
                code = (code << 1) | reader.read_bit()
                symbol = decode_table.get((width, code))
                if symbol is not None:
                    symbols.append(symbol)
                    matched = True
                    break
            if not matched:
                raise BitstreamError("bitstream payload contains an unknown code")

        return np.asarray(symbols, dtype=np.int32)
