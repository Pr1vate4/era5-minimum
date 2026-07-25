import pytest
import json
import numpy as np
import xarray as xr
from pathlib import Path
from era5_minimum.data.manifest import compute_manifest_sha256
from era5_minimum.data.validation import validate_zarr


@pytest.fixture
def mock_valid_zarr_and_manifest(tmp_path: Path):
    """Создает минимальный валидный Zarr и манифест для тестов."""
    zarr_path = tmp_path / "valid_data.zarr"

    # 2 шага времени, 28 каналов, сетка 4x4
    data = np.random.rand(2, 28, 4, 4).astype(np.float32)
    # Принудительно добавляем NaN в SST (канал 5) для имитации суши (требование контракта)
    data[:, 5, 0, 0] = np.nan

    ds = xr.Dataset({
        "data": (["time", "channel", "latitude", "longitude"], data)
    }, coords={
        "time": np.arange(2),
        "channel": np.arange(28),
        "latitude": np.arange(4),
        "longitude": np.arange(4)
    })
    ds.to_zarr(zarr_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_data = {
        "schema_version": "1.0.0",
        "integrity": {},
    }
    manifest_data["integrity"]["manifest_sha256"] = compute_manifest_sha256(
        manifest_data
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    return str(zarr_path), str(manifest_path)


def test_validate_zarr_success(mock_valid_zarr_and_manifest):
    """Валидация должна проходить для корректных данных."""
    zarr_path, manifest_path = mock_valid_zarr_and_manifest

    # Должно выполниться без исключений и вернуть True
    assert validate_zarr(zarr_path, manifest_path) is True


def test_validate_fails_on_wrong_channel_count(tmp_path: Path, monkeypatch):
    """Валидация должна падать, если каналов не 28."""
    zarr_path = tmp_path / "wrong_channels.zarr"
    ds = xr.Dataset({
        "data": (["time", "channel", "latitude", "longitude"], np.ones((2, 8, 4, 4), dtype=np.float32))
        # Только 8 каналов (моковый размер)
    })
    ds.to_zarr(zarr_path)

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"integrity": {"manifest_sha256": "dummy"}}, f)

    monkeypatch.setattr("era5_minimum.data.manifest.compute_sha256", lambda x: "dummy")

    with pytest.raises(AssertionError, match="channel"):
        validate_zarr(str(zarr_path), str(manifest_path))


def test_validate_fails_on_missing_sst_nan(tmp_path: Path, monkeypatch):
    """Валидация должна падать, если SST не содержит NaN (суша заполнена нулями, а не NaN)."""
    zarr_path = tmp_path / "no_nan_sst.zarr"
    # Заполняем всё единицами, включая SST (канал 5)
    ds = xr.Dataset({
        "data": (["time", "channel", "latitude", "longitude"], np.ones((2, 28, 4, 4), dtype=np.float32))
    }, coords={
        "time": np.arange(2),
        "channel": np.arange(28),
        "latitude": np.arange(4),
        "longitude": np.arange(4)
    })
    ds.to_zarr(zarr_path)

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"integrity": {"manifest_sha256": "dummy"}}, f)

    monkeypatch.setattr("era5_minimum.data.manifest.compute_sha256", lambda x: "dummy")

    with pytest.raises(AssertionError, match="SST должен содержать NaN над сушей"):
        validate_zarr(str(zarr_path), str(manifest_path))
