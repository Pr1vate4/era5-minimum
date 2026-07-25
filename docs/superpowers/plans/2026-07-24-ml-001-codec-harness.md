# ML-001 Codec Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать воспроизводимый ERA5 codec harness с детерминированной нормализацией, квантованием, canonical Huffman bitstream, exact symbol roundtrip и provenance-rich metadata.

**Architecture:** Добавляем новый `era5_minimum.codec` пакет, который держит нормализацию, квантование, entropy coding и I/O в отдельных файлах. Сначала работаем на plain NumPy tensors, чтобы честно проверить bitstream contract до интеграции с конкурсной моделью. Результат всегда отдельно считает tensor ratio и serialized ratio.

**Tech Stack:** Python 3.12, NumPy, stdlib (`dataclasses`, `heapq`, `json`, `hashlib`, `struct`), pytest, текущие project tools.

---

### Task 1: Codec types and scalar quantization

**Files:**
- Create: `src/era5_minimum/codec/__init__.py`
- Create: `src/era5_minimum/codec/types.py`
- Create: `src/era5_minimum/codec/normalization.py`
- Create: `src/era5_minimum/codec/quantization.py`
- Create: `tests/test_codec_quantization.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

Run: `pytest tests/test_codec_quantization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'era5_minimum.codec'`

- [ ] **Step 2: Write minimal implementation**

```python
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScalarQuantizer:
    step: float
    zero_point: int = 0

    def quantize(self, values: np.ndarray) -> np.ndarray:
        scaled = np.asarray(values, dtype=np.float32) / self.step
        return np.rint(scaled).astype(np.int32) + self.zero_point

    def dequantize(self, symbols: np.ndarray) -> np.ndarray:
        shifted = np.asarray(symbols, dtype=np.int32).astype(np.float32) - self.zero_point
        return shifted * self.step


@dataclass(frozen=True)
class NormalizationSpec:
    channel_order: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    source_manifest_sha256: str
    train_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_order": list(self.channel_order),
            "mean": np.asarray(self.mean, dtype=np.float32).tolist(),
            "std": np.asarray(self.std, dtype=np.float32).tolist(),
            "source_manifest_sha256": self.source_manifest_sha256,
            "train_only": self.train_only,
        }
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_codec_quantization.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/era5_minimum/codec/__init__.py src/era5_minimum/codec/types.py src/era5_minimum/codec/normalization.py src/era5_minimum/codec/quantization.py tests/test_codec_quantization.py
git commit -m "feat(codec): add quantization primitives"
```

### Task 2: Canonical Huffman bitstream

**Files:**
- Create: `src/era5_minimum/codec/huffman.py`
- Create: `src/era5_minimum/codec/bitstream.py`
- Create: `tests/test_codec_bitstream.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

Run: `pytest tests/test_codec_bitstream.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'era5_minimum.codec.bitstream'`

- [ ] **Step 2: Write minimal implementation**

```python
from dataclasses import dataclass
from collections import Counter
import json
import struct

import numpy as np


class BitstreamError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalHuffmanCoder:
    code_lengths: dict[int, int]
    codes: dict[int, tuple[int, int]]

    @classmethod
    def from_symbols(cls, symbols: np.ndarray) -> "CanonicalHuffmanCoder":
        counts = Counter(int(value) for value in np.asarray(symbols, dtype=np.int32).ravel())
        code_lengths = _build_code_lengths(counts)
        return cls(code_lengths=code_lengths, codes=_build_canonical_codes(code_lengths))

    def encode(self, symbols: np.ndarray) -> bytes:
        writer = _BitWriter()
        flat = np.asarray(symbols, dtype=np.int32).ravel()
        for value in flat:
            code, width = self.codes[int(value)]
            writer.write(code, width)
        header = json.dumps(
            {"code_lengths": self.code_lengths, "symbol_count": int(flat.size)}
        ).encode("utf-8")
        return struct.pack("<I", len(header)) + header + writer.finish()

    def decode(self, payload: bytes, symbol_count: int) -> np.ndarray:
        if len(payload) < 4:
            raise BitstreamError("bitstream is truncated")
        header_len = struct.unpack("<I", payload[:4])[0]
        if len(payload) < 4 + header_len:
            raise BitstreamError("bitstream header is truncated")
        raw = payload[4 + header_len :]
        reader = _BitReader(raw)
        symbols = [reader.read(self.codes, symbol_count) for _ in range(symbol_count)]
        if len(symbols) != symbol_count:
            raise BitstreamError("bitstream payload is truncated")
        return np.asarray(symbols, dtype=np.int32)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_codec_bitstream.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/era5_minimum/codec/huffman.py src/era5_minimum/codec/bitstream.py tests/test_codec_bitstream.py
git commit -m "feat(codec): add canonical bitstream"
```

### Task 3: Codec harness and metadata

**Files:**
- Create: `src/era5_minimum/codec/harness.py`
- Create: `tests/test_codec_harness.py`
- Modify: `docs/DECISIONS.md`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import numpy as np

from era5_minimum.codec.harness import CodecConfig, CodecHarness
from era5_minimum.codec.normalization import NormalizationSpec


def test_codec_harness_writes_metadata_and_separates_ratios(tmp_path: Path) -> None:
    tensor = np.array([[[[0.0, 0.25], [0.5, 0.75]]]], dtype=np.float32)
    config = CodecConfig(
        version="ml-001",
        channel_order=("t2m",),
        grid="0.5deg",
        quantization_step=0.25,
        seed=7,
        git_commit="deadbeef",
    )
    normalization = NormalizationSpec(
        channel_order=("t2m",),
        mean=np.array([0.0], dtype=np.float32),
        std=np.array([1.0], dtype=np.float32),
        source_manifest_sha256="abc123",
        train_only=True,
    )

    result = CodecHarness(config=config, normalization=normalization).encode_decode(tensor, tmp_path)

    assert result.roundtrip_ok is True
    assert result.tensor_ratio > 0
    assert result.serialized_ratio > 0
    assert result.metadata["normalization"]["source_manifest_sha256"] == "abc123"
```

Run: `pytest tests/test_codec_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'era5_minimum.codec.harness'`

- [ ] **Step 2: Write minimal implementation**

```python
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from era5_minimum.codec.bitstream import CanonicalHuffmanCoder
from era5_minimum.codec.normalization import NormalizationSpec
from era5_minimum.codec.quantization import ScalarQuantizer


@dataclass(frozen=True)
class CodecConfig:
    version: str
    channel_order: tuple[str, ...]
    grid: str
    quantization_step: float
    seed: int
    git_commit: str | None


@dataclass(frozen=True)
class CodecResult:
    bitstream_path: Path
    metadata_path: Path
    tensor_ratio: float
    serialized_ratio: float
    roundtrip_ok: bool
    metadata: dict[str, Any]


class CodecHarness:
    def __init__(self, config: CodecConfig, normalization: NormalizationSpec) -> None:
        self.config = config
        self.normalization = normalization
        self.quantizer = ScalarQuantizer(step=config.quantization_step)

    def encode_decode(self, tensor: np.ndarray, output_dir: Path) -> CodecResult:
        normalized = (np.asarray(tensor, dtype=np.float32) - self.normalization.mean.reshape(1, -1, 1, 1)) / self.normalization.std.reshape(1, -1, 1, 1)
        symbols = self.quantizer.quantize(normalized)
        coder = CanonicalHuffmanCoder.from_symbols(symbols.ravel())
        payload = coder.encode(symbols.ravel())
        decoded_symbols = coder.decode(payload, symbol_count=symbols.size).reshape(symbols.shape)
        roundtrip_ok = np.array_equal(decoded_symbols, symbols)
        output_dir.mkdir(parents=True, exist_ok=True)
        bitstream_path = output_dir / "codec.bin"
        metadata_path = output_dir / "codec.json"
        bitstream_path.write_bytes(payload)
        metadata = {
            "config": {
                "version": self.config.version,
                "channel_order": list(self.config.channel_order),
                "grid": self.config.grid,
                "quantization_step": self.config.quantization_step,
                "seed": self.config.seed,
                "git_commit": self.config.git_commit,
            },
            "normalization": self.normalization.to_dict(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return CodecResult(
            bitstream_path=bitstream_path,
            metadata_path=metadata_path,
            tensor_ratio=float(tensor.size / max(symbols.size, 1)),
            serialized_ratio=float(tensor.nbytes / max(len(payload), 1)),
            roundtrip_ok=roundtrip_ok,
            metadata=metadata,
        )
```

- [ ] **Step 3: Update docs with the codec decision**

Append to `docs/DECISIONS.md` a short note that the first ML-001 codec step uses per-channel scalar quantization plus canonical Huffman because it gives exact symbol roundtrip, honest serialized ratio reporting, and no new dependencies.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codec_harness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/era5_minimum/codec/harness.py tests/test_codec_harness.py docs/DECISIONS.md
git commit -m "feat(codec): add codec harness metadata"
```

### Task 4: Verification and release cleanup

**Files:**
- Modify if needed: `src/era5_minimum/codec/*.py`
- Modify if needed: `tests/test_codec_*.py`
- No new files unless verification exposes a real defect

- [ ] **Step 1: Run the codec-focused tests**

Run: `pytest tests/test_codec_quantization.py tests/test_codec_bitstream.py tests/test_codec_harness.py -v`
Expected: PASS

- [ ] **Step 2: Run the repository verification command**

Run: `make verify`
Expected: PASS

- [ ] **Step 3: Check that metadata and ratio semantics stayed honest**

Run: `git diff --check && rg -n "canonical Huffman|tensor ratio|serialized ratio" docs/DECISIONS.md src/era5_minimum/codec`
Expected: no whitespace errors and the codec decision text is present; no claim that tensor ratio equals serialized ratio.

- [ ] **Step 4: Commit the final verification fixups**

```bash
git add src/era5_minimum/codec tests/test_codec_*.py docs/DECISIONS.md
git commit -m "feat(codec): finish harness verification"
```
