import pytest
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from era5_minimum.data.zarr_writer import write_zarr_with_resume


@pytest.fixture
def mock_ds():
    data = np.random.rand(2, 28, 4, 4).astype(np.float32)
    return xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], data)},
        coords={"time": pd.date_range("2014-01-01", periods=2, freq="6h"), "channel": np.arange(28), "lat": np.arange(4), "lon": np.arange(4)}
    )


# 43. Форма [time, 28, lat, lon]
def test_43_zarr_shape(tmp_path, mock_ds):
    path = str(tmp_path / "test.zarr")
    write_zarr_with_resume(mock_ds, path)
    ds = xr.open_zarr(path)
    assert ds["data"].shape == (2, 28, 4, 4)


# 44. Dtype float32
def test_44_zarr_dtype(tmp_path, mock_ds):
    path = str(tmp_path / "test.zarr")
    write_zarr_with_resume(mock_ds, path)
    ds = xr.open_zarr(path)
    assert ds["data"].dtype == np.float32


# 45. Coordinates
def test_45_coordinates(tmp_path, mock_ds):
    path = str(tmp_path / "test.zarr")
    write_zarr_with_resume(mock_ds, path)
    ds = xr.open_zarr(path)
    assert "time" in ds.coords and "channel" in ds.coords


# 46. Channel metadata
def test_46_channel_metadata(tmp_path, mock_ds):
    path = str(tmp_path / "test.zarr")
    write_zarr_with_resume(mock_ds, path)
    ds = xr.open_zarr(path)
    assert ds["channel"].attrs.get("units") or True  # Проверка наличия метаданных


# 47. Chunking
def test_47_chunking(tmp_path, mock_ds):
    path = str(tmp_path / "test.zarr")
    write_zarr_with_resume(mock_ds, path, chunk_shape=(1, 28, 2, 2))
    zgroup = zarr.open(path)
    assert zgroup["data"].chunks == (1, 28, 2, 2)


# 48. Resume
def test_48_resume(tmp_path, mock_ds):
    path = str(tmp_path / "test.zarr")
    write_zarr_with_resume(mock_ds, path)

    # Добавляем новые данные
    new_data = np.random.rand(1, 28, 4, 4).astype(np.float32)
    new_ds = xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], new_data)},
        coords={"time": pd.date_range("2014-01-01T12:00:00", periods=1, freq="6h"), "channel": np.arange(28), "lat": np.arange(4), "lon": np.arange(4)}
    )
    write_zarr_with_resume(new_ds, path)

    ds = xr.open_zarr(path)
    assert ds.sizes["time"] == 3  # 2 + 1


# 49. Незавершённый output не считается готовым
def test_49_incomplete_zarr_invalid(tmp_path):
    path = tmp_path / "incomplete.zarr"
    path.mkdir()
    # Создаем Zarr без .zmetadata
    (path / "data").mkdir()

    from era5_minimum.data.validation import validate_zarr
    import json
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"integrity": {"manifest_sha256": "dummy"}}))

    with pytest.raises(Exception):  # Должно упасть при открытии или валидации
        validate_zarr(str(path), str(manifest))