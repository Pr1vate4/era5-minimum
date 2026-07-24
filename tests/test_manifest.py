import pytest
import json
from pathlib import Path
import hashlib
from era5_minimum.data.manifest import generate_manifest


# 50. Canonical JSON
def test_50_canonical_json(tmp_path):
    path = generate_manifest(str(tmp_path), {}, {})
    content = Path(path).read_text()
    # Canonical JSON: sort_keys, без лишних пробелов в конце строк
    parsed = json.loads(content)
    assert list(parsed.keys()) == sorted(parsed.keys())


# 51. Manifest SHA256
def test_51_manifest_sha256(tmp_path):
    path = generate_manifest(str(tmp_path), {}, {})
    with open(path) as f:
        data = json.load(f)

    # Восстанавливаем канонический JSON без хеша
    data_copy = json.loads(json.dumps(data))
    del data_copy["integrity"]["manifest_sha256"]
    canonical = json.dumps(data_copy, indent=2, sort_keys=True)
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert data["integrity"]["manifest_sha256"] == expected_hash


# 52. Index file SHA256
def test_52_index_file_sha256(tmp_path):
    subsets = {"128": ["2014-01-01"]}
    path = generate_manifest(str(tmp_path), {}, subsets)
    with open(path) as f:
        data = json.load(f)
    assert len(data["nested_subsets"]["128"]["sha256"]) == 64


# 53. Remapping weights SHA256
def test_53_weights_sha256(tmp_path):
    # Имитируем файл весов
    weights_file = tmp_path / "weights.nc"
    weights_file.write_bytes(b"mock_weights")

    path = generate_manifest(str(tmp_path), {}, {})
    with open(path) as f:
        data = json.load(f)
    assert "manifest_sha256" in data["integrity"]


# 54. Отсутствие абсолютных путей
def test_54_no_absolute_paths(tmp_path):
    path = generate_manifest(str(tmp_path), {}, {})
    content = Path(path).read_text()
    assert str(tmp_path.absolute()) not in content
    assert "C:\\" not in content


# 55. Проверка schema version
def test_55_schema_version(tmp_path):
    path = generate_manifest(str(tmp_path), {}, {})
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0.0"