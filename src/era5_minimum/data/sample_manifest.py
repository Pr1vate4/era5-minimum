from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .seasonal_subsets import generate_nested_subsets
from .selection import get_split_timestamps
from .weatherbench2 import WB2_URL

MANIFEST_SCHEMA_VERSION = "1.0.0"
TEMPORAL_EMBARGO_HOURS = 168
DEFAULT_SAMPLE_SIZES = (16, 32, 64, 128)


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a manifest payload with stable key and separator ordering."""
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _split_bounds() -> dict[str, dict[str, str]]:
    return {
        split: {
            "start": get_split_timestamps(split).min().strftime("%Y-%m-%dT%H:%M:%S"),
            "end": get_split_timestamps(split).max().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for split in ("train", "val", "test")
    }


@dataclass(frozen=True)
class TrainingSampleManifest:
    """Immutable provenance and timestamp selection description for training."""

    seed: int
    temporal_embargo_hours: int
    source_uri: str
    subsets: dict[int, tuple[str, ...]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe manifest with a self-excluding SHA-256 digest."""
        sizes = sorted(self.subsets)
        nested_subsets: dict[str, dict[str, object]] = {}
        for index, size in enumerate(sizes):
            timestamps = self.subsets[size]
            nested_subsets[str(size)] = {
                "count": len(timestamps),
                "parent_size": sizes[index - 1] if index else None,
                "timestamps": list(timestamps),
            }
        payload: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "train_only": True,
            "seed": self.seed,
            "temporal_embargo_hours": self.temporal_embargo_hours,
            "source_uri": self.source_uri,
            "split_bounds": _split_bounds(),
            "subsets": nested_subsets,
            "integrity": {"hash_algorithm": "sha256"},
        }
        payload["integrity"]["manifest_sha256"] = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
        return payload


def write_training_sample_manifest(
    path: Path,
    *,
    sizes: tuple[int, ...] = DEFAULT_SAMPLE_SIZES,
    seed: int = 42,
) -> Path:
    """Write a deterministic leakage-safe nested training sample manifest."""
    generated = generate_nested_subsets(sizes, seed=seed)
    manifest = TrainingSampleManifest(
        seed=seed,
        temporal_embargo_hours=TEMPORAL_EMBARGO_HOURS,
        source_uri=WB2_URL,
        subsets={size: tuple(values) for size, values in generated.items()},
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(manifest.to_dict()) + b"\n")
    return output
