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

    # 1. Формируем канонический JSON БЕЗ поля manifest_sha256
    canonical_json = json.dumps(manifest, indent=2, sort_keys=True)

    # 2. Считаем хеш этого "чистого" JSON
    manifest_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    # 3. Добавляем хеш в словарь
    manifest["integrity"]["manifest_sha256"] = manifest_hash

    # 4. Записываем финальный файл (уже с хешем внутри)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return str(path)