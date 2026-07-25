import json
import hashlib
import os
from pathlib import Path
from .channel_spec import CHANNEL_SPEC


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest_sha256(manifest: dict) -> str:
    """Hash the canonical manifest content without its self-referential hash."""
    manifest_copy = json.loads(json.dumps(manifest))
    integrity = manifest_copy.setdefault("integrity", {})
    integrity.pop("manifest_sha256", None)
    canonical_json = json.dumps(manifest_copy, indent=2, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def generate_manifest(output_dir: str, stats: dict, subsets: dict, weights_path: str = None) -> str:
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "era5-minimum-28ch-real",
        "channel_spec": [{"index": c.index, "name": c.name, "units": c.units} for c in CHANNEL_SPEC],
        "splits": {"train": "2014-2019", "val": "2020", "test": "2021"},
        "nested_subsets": {
            str(k): {"count": len(v), "sha256": hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()}
            for k, v in subsets.items()
        },
        "statistics": stats,
        "integrity": {"storage_compression_is_not_codec_result": True}
    }

    if weights_path and os.path.exists(weights_path):
        manifest["integrity"]["weights_sha256"] = compute_sha256(weights_path)

    path = Path(output_dir) / "manifest.json"

    # Добавляем digest канонического JSON без self-referential hash.
    manifest["integrity"]["manifest_sha256"] = compute_manifest_sha256(manifest)

    # Записываем финальный файл (уже с хешем внутри).
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return str(path)
