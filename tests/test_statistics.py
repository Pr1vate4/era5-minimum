import pytest
import numpy as np
import xarray as xr
from era5_minimum.data.statistics import compute_train_statistics


@pytest.fixture
def chunked_train_ds():
    data = np.array([[[[10.0, 20.0], [np.nan, 40.0]]]], dtype=np.float32)
    ds = xr.Dataset({"t2m": (["time", "channel", "latitude", "longitude"], data)})
    return ds.chunk({"time": 1, "channel": 1, "latitude": 2, "longitude": 2})


def test_35_mean_train_only(chunked_train_ds):
    stats = compute_train_statistics(chunked_train_ds)
    assert isinstance(stats["t2m"]["mean"], float)  # Проверяем, что это нативный float
    assert stats["t2m"]["mean"] == pytest.approx(23.333, rel=1e-2)


def test_36_std_train_only(chunked_train_ds):
    stats = compute_train_statistics(chunked_train_ds)
    assert isinstance(stats["t2m"]["std"], float)
    assert stats["t2m"]["std"] > 0


def test_37_test_no_influence(chunked_train_ds):
    stats1 = compute_train_statistics(chunked_train_ds)

    # Создаем абсолютно новый датасет с другими данными, а не мутируем старый
    data_polluted = np.array([[[[10000.0, 20.0], [np.nan, 40.0]]]], dtype=np.float32)
    ds_polluted = xr.Dataset({"t2m": (["time", "channel", "latitude", "longitude"], data_polluted)})
    ds_polluted = ds_polluted.chunk({"time": 1, "channel": 1, "latitude": 2, "longitude": 2})

    stats2 = compute_train_statistics(ds_polluted)

    mean1 = stats1["t2m"]["mean"]
    mean2 = stats2["t2m"]["mean"]

    # Если функция вернула список (для 28 каналов), берем первый элемент
    if isinstance(mean1, list): mean1 = mean1[0]
    if isinstance(mean2, list): mean2 = mean2[0]

    assert mean1 != mean2, "Изменение данных должно влиять на статистику"


def test_38_sst_land_no_influence():
    data_with_nan = np.array([[[[300.0, np.nan]]]], dtype=np.float32)
    data_no_nan = np.array([[[[300.0, 300.0]]]], dtype=np.float32)
    ds1 = xr.Dataset({"sst": (["time", "channel", "latitude", "longitude"], data_with_nan)}).chunk()
    ds2 = xr.Dataset({"sst": (["time", "channel", "latitude", "longitude"], data_no_nan)}).chunk()

    s1 = compute_train_statistics(ds1)
    s2 = compute_train_statistics(ds2)
    assert s1["sst"]["mean"] == pytest.approx(s2["sst"]["mean"])


def test_39_nan_excluded(chunked_train_ds):
    stats = compute_train_statistics(chunked_train_ds)
    assert not np.isnan(stats["t2m"]["mean"])
    assert not np.isnan(stats["t2m"]["std"])


def test_40_valid_count(chunked_train_ds):
    stats = compute_train_statistics(chunked_train_ds)
    assert isinstance(stats["t2m"]["valid_count"], int)  # Проверяем, что это нативный int
    assert stats["t2m"]["valid_count"] == 3


def test_41_percentiles(chunked_train_ds):
    stats = compute_train_statistics(chunked_train_ds)
    assert "p05" in stats["t2m"] and "p995" in stats["t2m"]
    assert stats["t2m"]["p05"] <= stats["t2m"]["p995"]


def test_42_reproducibility(chunked_train_ds):
    s1 = compute_train_statistics(chunked_train_ds)
    s2 = compute_train_statistics(chunked_train_ds)
    assert s1["t2m"]["mean"] == s2["t2m"]["mean"]