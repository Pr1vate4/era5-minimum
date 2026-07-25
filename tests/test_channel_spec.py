import pytest
import numpy as np
import xarray as xr
from era5_minimum.data.channel_spec import CHANNEL_SPEC, validate_wb2_compatibility

# 1. Ровно 28 каналов
def test_01_exact_channel_count():
    assert len(CHANNEL_SPEC) == 28

# 2. Строгий порядок
def test_02_strict_order():
    expected_order = ["t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc"] + \
                     [f"{var}{lvl}" for var in ["t", "u", "v", "z", "q"] for lvl in [1000, 925, 850, 700]]
    actual_order = [c.name for c in CHANNEL_SPEC]
    assert actual_order == expected_order

# 3. Правильные WeatherBench2 names
@pytest.mark.parametrize("idx,wb2_name", [
    (0, "2m_temperature"), (1, "mean_sea_level_pressure"), (4, "total_precipitation_6hr"),
    (8, "temperature_1000hPa"), (27, "specific_humidity_700hPa")
])
def test_03_wb2_names(idx, wb2_name):
    assert CHANNEL_SPEC[idx].wb2_name == wb2_name

# 4. Правильные pressure levels
def test_04_pressure_levels():
    pressure_channels = [c for c in CHANNEL_SPEC if c.type == "pressure"]
    levels = [c.level for c in pressure_channels]
    assert levels == [1000, 925, 850, 700] * 5


# 5. Ошибка при отсутствии переменной
def test_05_missing_variable_raises():
    """Проверяем, что функция бросает ValueError при отсутствии нужной переменной."""
    # Создаем мок-датасет только с одной переменной (не хватает остальных 27)
    mock_ds = xr.Dataset({
        "2m_temperature": (["time", "latitude", "longitude"], np.random.rand(2, 4, 4)),
        "latitude": (["latitude"], np.arange(4)),
        "longitude": (["longitude"], np.arange(4)),
        "time": (["time"], np.arange(2))
    })

    with pytest.raises(ValueError, match="Missing variable"):
        validate_wb2_compatibility(mock_ds, CHANNEL_SPEC)


# 6. Ошибка при отсутствии level
def test_06_missing_level_raises():
    """Проверяем, что функция бросает ValueError при отсутствии нужного уровня давления."""
    # Создаем мок-датасет с temperature, но только на уровнях 1000 и 850 (не хватает 925 и 700)
    mock_ds = xr.Dataset({
        "temperature": (["time", "level", "latitude", "longitude"], np.random.rand(2, 2, 4, 4)),
        "level": (["level"], [1000, 850]),  # Отсутствуют 925 и 700
        "latitude": (["latitude"], np.arange(4)),
        "longitude": (["longitude"], np.arange(4)),
        "time": (["time"], np.arange(2))
    })

    # Также нужно добавить остальные surface-переменные, чтобы тест не упал на проверке переменных
    for ch in CHANNEL_SPEC:
        if ch.type == "surface":
            mock_ds[ch.wb2_name] = (["time", "latitude", "longitude"], np.random.rand(2, 4, 4))

    with pytest.raises(ValueError, match="[Mm]issing.*level"):
        validate_wb2_compatibility(mock_ds, CHANNEL_SPEC)

# 7. Ошибка при tp1h вместо tp6h
def test_07_tp1h_forbidden():
    names = [c.wb2_name for c in CHANNEL_SPEC]
    assert "total_precipitation" in names or "total_precipitation_6hr" in names
    assert "total_precipitation_1hr" not in names
    assert any(c.name == "tp6h" for c in CHANNEL_SPEC)

# 8. Проверка units
def test_08_units_correctness():
    units_map = {c.name: c.units for c in CHANNEL_SPEC}
    assert units_map["t2m"] == "K"
    assert units_map["mslp"] == "Pa"
    assert units_map["tp6h"] == "m"
    assert units_map["z1000"] == "m2 s-2"