import pytest
import pandas as pd
from era5_minimum.data.seasonal_subsets import generate_nested_subsets, month_to_season
from era5_minimum.data.selection import get_split_timestamps


def _count_seasons(timestamps: list[str]) -> dict:
    """Хелпер для подсчета сезонов в списке timestamp'ов."""
    months = pd.to_datetime(timestamps).month
    seasons = [month_to_season(m) for m in months]
    counts = {"DJF": 0, "MAM": 0, "JJA": 0, "SON": 0}
    for s in seasons:
        counts[s] += 1
    return counts


def test_strict_nesting():
    """Контракт: subset_128 ⊂ subset_256 ⊂ ... ⊂ subset_8192."""
    subsets = generate_nested_subsets(n_list=[128, 256, 512], seed=42)

    set_128 = set(subsets[128])
    set_256 = set(subsets[256])
    set_512 = set(subsets[512])

    assert set_128.issubset(set_256), "N=128 не является подмножеством N=256"
    assert set_256.issubset(set_512), "N=256 не является подмножеством N=512"
    assert len(set_128) == 128
    assert len(set_256) == 256


def test_seasonal_balance_round_robin():
    """Контракт: Ровное количество кадров из сезонов DJF, MAM, JJA, SON."""
    subsets = generate_nested_subsets(n_list=[128, 256], seed=42)

    counts_128 = _count_seasons(subsets[128])
    assert counts_128 == {"DJF": 32, "MAM": 32, "JJA": 32, "SON": 32}, f"Дисбаланс для N=128: {counts_128}"

    counts_256 = _count_seasons(subsets[256])
    assert counts_256 == {"DJF": 64, "MAM": 64, "JJA": 64, "SON": 64}, f"Дисбаланс для N=256: {counts_256}"


def test_train_pool_isolation():
    """Контракт: В наборы входят только timestamps из Train pool (2014-2019)."""
    subsets = generate_nested_subsets(n_list=[128], seed=42)
    timestamps = pd.to_datetime(subsets[128])

    assert timestamps.min().year >= 2014, "Найдены данные до 2014 года"
    assert timestamps.max().year <= 2019, "Найдены данные после 2019 года (Data Leakage из Val/Test)"


def test_deterministic_seed():
    """Контракт: Генерация через фиксированный seed должна быть детерминированной."""
    subsets_1 = generate_nested_subsets(n_list=[128], seed=42)
    subsets_2 = generate_nested_subsets(n_list=[128], seed=42)

    assert subsets_1[128] == subsets_2[128], "Результат недетерминирован при одинаковом seed"


def test_train_end_respects_validation_embargo():
    """Train заканчивается до validation с семидневным временным зазором."""
    train = get_split_timestamps("train")
    validation = get_split_timestamps("val")

    assert train.max() == pd.Timestamp("2019-12-24T18:00:00")
    assert validation.min() == pd.Timestamp("2020-01-01T00:00:00")
    assert validation.min() - train.max() >= pd.Timedelta(days=7)
