import hashlib
import json
from pathlib import Path

from era5_minimum.data.sample_manifest import (
    TrainingSampleManifest,
    write_training_sample_manifest,
)


def _payload_without_hash(payload: dict[str, object]) -> dict[str, object]:
    unsigned = dict(payload)
    integrity = dict(unsigned["integrity"])  # type: ignore[index]
    integrity.pop("manifest_sha256")
    unsigned["integrity"] = integrity
    return unsigned


def test_manifest_contains_nested_timestamps_and_provenance(tmp_path: Path):
    output = write_training_sample_manifest(tmp_path / "samples.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["train_only"] is True
    assert payload["seed"] == 42
    assert payload["temporal_embargo_hours"] == 168
    assert payload["split_bounds"]["train"]["end"] == "2019-12-24T18:00:00"
    assert payload["split_bounds"]["val"]["start"] == "2020-01-01T00:00:00"
    assert payload["subsets"]["16"]["parent_size"] is None
    assert payload["subsets"]["32"]["parent_size"] == 16
    assert payload["subsets"]["128"]["parent_size"] == 64
    assert len(payload["subsets"]["128"]["timestamps"]) == 128


def test_manifest_hash_is_canonical_and_deterministic(tmp_path: Path):
    first = write_training_sample_manifest(tmp_path / "first.json", seed=42)
    second = write_training_sample_manifest(tmp_path / "second.json", seed=42)

    assert first.read_bytes() == second.read_bytes()

    payload = json.loads(first.read_text(encoding="utf-8"))
    canonical = json.dumps(
        _payload_without_hash(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_hash = hashlib.sha256(canonical).hexdigest()
    assert payload["integrity"]["manifest_sha256"] == expected_hash


def test_manifest_dataclass_serializes_tuple_timestamps():
    manifest = TrainingSampleManifest(
        seed=7,
        temporal_embargo_hours=168,
        source_uri="memory://era5",
        subsets={16: ("2014-01-01T00:00:00",) * 16},
    )

    payload = manifest.to_dict()

    assert payload["source_uri"] == "memory://era5"
    assert payload["subsets"]["16"]["timestamps"][0] == "2014-01-01T00:00:00"
    assert payload["subsets"]["16"]["parent_size"] is None
