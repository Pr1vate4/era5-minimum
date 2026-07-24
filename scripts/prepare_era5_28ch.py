import click
import xarray as xr
import numpy as np
import pandas as pd
import shutil
import sys
import os
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from era5_minimum.data.channel_spec import CHANNEL_SPEC, validate_wb2_compatibility
from era5_minimum.data.weatherbench2 import open_wb2_era5
from era5_minimum.data.selection import SPLITS
from era5_minimum.data.seasonal_subsets import generate_nested_subsets
from era5_minimum.data.grids import get_target_grid_05
from era5_minimum.data.remapping import conservative_remap
from era5_minimum.data.masks import apply_masked_sst_remap
from era5_minimum.data.statistics import compute_train_statistics
from era5_minimum.data.zarr_writer import write_zarr_with_resume
from era5_minimum.data.manifest import generate_manifest
from era5_minimum.data.validation import validate_zarr


@click.group()
def cli():
    """ERA5-Minimum 28ch Data Pipeline"""
    pass


@cli.command()
def smoke():
    """Генерирует синтетический WeatherBench2-подобный dataset и прогоняет весь пайплайн локально."""
    print("🚀 Запуск Smoke-теста...")
    rng = np.random.default_rng(42)

    out_dir = Path(os.environ.get("SMOKE_OUTPUT_DIR", "smoke_test_output"))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Создаём synthetic WeatherBench2-подобный dataset
    print("📦 Создание synthetic WeatherBench2-подобного dataset...")

    # Временной диапазон: 4 сезона (по 10 дней каждый), 6-часовой шаг
    # Это даёт 4 × 10 × 4 = 160 timestamps, покрывающих все 4 сезона
    times = []
    for month in [1, 4, 7, 10]:  # DJF, MAM, JJA, SON
        month_times = pd.date_range(f"2014-{month:02d}-01", periods=40, freq="6h")
        times.extend(month_times)
    times = sorted(times)

    # Маленькая source grid: 10×20
    lat_in = np.linspace(-90, 90, 10)
    lon_in = np.linspace(0, 359, 20)

    # Ocean mask: левая половина — океан, правая — суша
    ocean_fraction = xr.DataArray(
        np.ones((10, 20), dtype=np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": lat_in, "longitude": lon_in}
    )
    ocean_fraction.values[:, 10:] = 0.0  # Правая половина — суша

    # Создаём переменные WeatherBench2
    data_vars = {}

    # 8 Surface variables
    surface_vars = [
        ("2m_temperature", "K"),
        ("mean_sea_level_pressure", "Pa"),
        ("10m_u_component_of_wind", "m s-1"),
        ("10m_v_component_of_wind", "m s-1"),
        ("total_precipitation_6hr", "m"),
        ("sea_surface_temperature", "K"),
        ("total_column_water_vapour", "kg m-2"),
        ("total_cloud_cover", "0-1"),
    ]

    for wb2_name, units in surface_vars:
        data = rng.random((len(times), 10, 20), dtype=np.float32) * 100

        # Для SST: над сушей ставим NaN
        if wb2_name == "sea_surface_temperature":
            data[:, ocean_fraction.values == 0] = np.nan

        data_vars[wb2_name] = (["time", "latitude", "longitude"], data)

    # 5 Atmospheric variables × 4 levels = 20 channels
    levels = [1000, 925, 850, 700]
    atmospheric_vars = [
        ("temperature", "K"),
        ("u_component_of_wind", "m s-1"),
        ("v_component_of_wind", "m s-1"),
        ("geopotential", "m2 s-2"),
        ("specific_humidity", "kg kg-1"),
    ]

    for wb2_name, units in atmospheric_vars:
        data = rng.random(
            (len(times), 4, 10, 20), dtype=np.float32
        ) * 100
        data_vars[wb2_name] = (["time", "level", "latitude", "longitude"], data)

    ds_wb2 = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "latitude": lat_in,
            "longitude": lon_in,
            "level": levels,
        }
    )

    print(
        f"✅ Synthetic dataset создан: {len(times)} timestamps, {len(lat_in)}×{len(lon_in)} grid, 8 surface + 20 atmospheric vars")

    # 2. Создаём target grid (меньше source)
    print("🔄 Создание target grid...")
    lat_out = np.linspace(-89, 89, 5)
    # Keep every target longitude inside the source-domain bounds. A value of
    # 359.5 would be outside the final source coordinate (359.0) and would
    # introduce interpolation NaNs into non-SST channels.
    lon_out = np.linspace(0.5, 358.5, 10)
    ds_target_grid = xr.Dataset({
        "latitude": (["latitude"], lat_out),
        "longitude": (["longitude"], lon_out),
    })

    # 3. Remapping через scipy (упрощённый, не conservative, но для smoke достаточно)
    print("🔄 Remapping через scipy.interpolate...")
    from scipy.interpolate import RegularGridInterpolator

    remapped_vars = {}
    for var in ds_wb2.data_vars:
        data_in = ds_wb2[var].values
        dims = ds_wb2[var].dims

        if "level" in dims:
            # Atmospheric variable
            data_out = np.zeros((len(times), 4, len(lat_out), len(lon_out)), dtype=np.float32)
            for t in range(len(times)):
                for l in range(4):
                    interp = RegularGridInterpolator(
                        (lat_in, lon_in),
                        data_in[t, l],
                        method='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    lat_grid, lon_grid = np.meshgrid(lat_out, lon_out, indexing='ij')
                    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
                    data_out[t, l] = interp(points).reshape(len(lat_out), len(lon_out))
            remapped_vars[var] = (["time", "level", "latitude", "longitude"], data_out)
        else:
            # Surface variable
            data_out = np.zeros((len(times), len(lat_out), len(lon_out)), dtype=np.float32)
            for t in range(len(times)):
                interp = RegularGridInterpolator(
                    (lat_in, lon_in),
                    data_in[t],
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                lat_grid, lon_grid = np.meshgrid(lat_out, lon_out, indexing='ij')
                points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
                data_out[t] = interp(points).reshape(len(lat_out), len(lon_out))
            remapped_vars[var] = (["time", "latitude", "longitude"], data_out)

    ds_remapped = xr.Dataset(
        remapped_vars,
        coords={
            "time": times,
            "latitude": lat_out,
            "longitude": lon_out,
            "level": levels,
        }
    )

    # 4. Masked remapping для SST
    print("🌊 Masked remapping для SST...")
    interp_mask = RegularGridInterpolator(
        (lat_in, lon_in),
        ocean_fraction.values,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    lat_grid, lon_grid = np.meshgrid(lat_out, lon_out, indexing='ij')
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    ocean_fraction_out = interp_mask(points).reshape(len(lat_out), len(lon_out))

    sst_remapped = ds_remapped["sea_surface_temperature"].values.copy()
    sst_remapped[:, ocean_fraction_out < 0.5] = np.nan
    ds_remapped["sea_surface_temperature"] = (["time", "latitude", "longitude"], sst_remapped)

    # 5. Формируем 28 каналов в строгом порядке
    print("📊 Формирование 28 каналов...")
    data_list = []
    for ch in CHANNEL_SPEC:
        if ch.type == "surface":
            var_data = ds_remapped[ch.wb2_name]
        else:
            var_name = ch.wb2_name.rsplit("_", 1)[0]
            var_data = ds_remapped[var_name].sel(level=ch.level)
            # ИСПРАВЛЕНИЕ: удаляем scalar coordinate level, чтобы избежать конфликта при concat
            var_data = var_data.drop_vars("level", errors="ignore")
        data_list.append(var_data)

    data_array = xr.concat(data_list, dim="channel")
    data_array = data_array.transpose("time", "channel", "latitude", "longitude")
    data_array = data_array.astype(np.float32)

    ds_out = xr.Dataset(
        {"data": data_array},
        coords={
            "time": times,
            "channel": np.arange(28),
            "latitude": lat_out,
            "longitude": lon_out,
        }
    )

    print(f"✅ 28 каналов сформированы: shape={data_array.shape}")

    # 6. Записываем в Zarr
    print("💾 Запись в Zarr...")
    zarr_path = out_dir / "data.zarr"
    ds_out.to_zarr(zarr_path, mode="w")
    print(f"✅ Zarr создан: {zarr_path}")

    # 7. Создаём вложенные subsets
    print("📑 Создание вложенных subsets...")
    subsets = generate_nested_subsets(n_list=[4, 8, 16], seed=42)
    print(f"✅ Subsets созданы: {list(subsets.keys())}")

    # 8. Считаем train-only statistics
    print("📈 Подсчёт train-only statistics...")
    stats = compute_train_statistics(ds_out)
    print("✅ Statistics посчитаны")

    # 9. Создаём manifest с SHA256
    print("📝 Создание manifest...")
    manifest_path = generate_manifest(str(out_dir), stats, subsets)
    print(f"✅ Manifest создан: {manifest_path}")

    # 10. Валидация
    print("🔍 Валидация...")
    validate_zarr(str(zarr_path), manifest_path)
    print("🎉 Smoke-тест успешно завершен!")


@cli.command()
def inspect():
    """Показать метаданные WeatherBench2 ERA5 (variables, dimensions, units, levels, time range)."""
    print("🔍 Открытие WeatherBench2 ERA5...")
    try:
        ds = open_wb2_era5(time_slice=slice("2020-01-01", "2020-01-02"), validate=False)

        print("\n=== Dimensions ===")
        for dim, size in ds.dims.items():
            print(f"  {dim}: {size}")

        print("\n=== Variables ===")
        for var in ds.data_vars:
            attrs = ds[var].attrs
            units = attrs.get("units", "N/A")
            print(f"  {var}: units={units}")

        print("\n=== Pressure Levels ===")
        if "level" in ds.coords:
            print(f"  {ds.coords['level'].values.tolist()}")

        print("\n=== Time Range ===")
        print(f"  {ds.time.values[0]} — {ds.time.values[-1]}")

        print("\n=== Coordinates ===")
        for coord in ds.coords:
            print(f"  {coord}: shape={ds.coords[coord].shape}")

    except Exception as e:
        print(f"❌ Ошибка при открытии WeatherBench2: {e}")
        print("Проверьте доступ к интернету и GCS.")
        sys.exit(1)


@cli.command()
@click.option("--config", default="configs/data/era5_28ch_full.yaml", help="Путь к YAML-конфигу")
def build_indices(config):
    """Сгенерировать вложенные сезонно-сбалансированные наборы (nested subsets)."""
    print("📊 Генерация индексов подвыборок...")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    n_list = cfg.get("subsets", {}).get("n_list", [128, 256, 512, 1024, 2048, 4096, 8192])
    seed = cfg.get("subsets", {}).get("seed", 42)

    subsets = generate_nested_subsets(n_list=n_list, seed=seed)

    output_dir = Path(cfg["storage"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for n, timestamps in subsets.items():
        idx_path = output_dir / f"subset_{n}.json"
        import json
        with open(idx_path, "w") as f:
            json.dump(timestamps, f, indent=2)
        print(f"✅ Создан {idx_path} ({len(timestamps)} timestamps)")

    print("✅ Индексы сгенерированы.")


@cli.command()
@click.option("--config", default="configs/data/era5_28ch_full.yaml", help="Путь к YAML-конфигу")
@click.option("--split", default="train", help="train, val, test")
def prepare(config, split):
    """Основная команда подготовки реальных данных из WeatherBench2."""
    print(f"🚀 Запуск подготовки для сплита: {split}")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["storage"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / f"{split}.zarr"

    # 1. Открытие WeatherBench2
    print("📥 Открытие WeatherBench2 ERA5...")
    time_slice = SPLITS[split]
    ds_wb2 = open_wb2_era5(time_slice=time_slice, validate=True)

    # 2. Ремаппинг на 0.5° сетку
    print("🔄 Conservative remapping на 0.5° сетку...")
    ds_target_grid = get_target_grid_05()

    # Ремаппинг всех переменных через инкапсулированную функцию
    ds_remapped = conservative_remap(ds_wb2, ds_target_grid)

    # 3. Формирование массива [time, channel, lat, lon]
    print("📦 Формирование массива с 28 каналами...")
    data_list = []
    for ch in CHANNEL_SPEC:
        if ch.type == "surface":
            var_data = ds_remapped[ch.wb2_name]
        else:
            var_name = ch.wb2_name.split("_")[0]  # temperature, u_component_of_wind, etc.
            var_data = ds_remapped[var_name].sel(level=ch.level)
        data_list.append(var_data)

    data_array = xr.concat(data_list, dim="channel")
    data_array = data_array.transpose("time", "channel", "latitude", "longitude")
    data_array = data_array.astype(np.float32)

    ds_out = xr.Dataset(
        {"data": data_array},
        coords={
            "time": ds_remapped.time,
            "channel": np.arange(28),
            "latitude": ds_target_grid.latitude,
            "longitude": ds_target_grid.longitude
        }
    )

    # 4. Запись в Zarr с resume
    print(f"💾 Запись в {zarr_path}...")
    chunk_shape = tuple(cfg["dataset"]["chunk_shape"])
    write_zarr_with_resume(ds_out, str(zarr_path), chunk_shape=chunk_shape)

    # 5. Статистики (только для train)
    if split == "train":
        print("📊 Подсчёт статистик...")
        stats = compute_train_statistics(ds_out)
        stats_path = output_dir / "statistics.json"
        import json
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Статистики сохранены в {stats_path}")
    else:
        stats = {}

    # 6. Манифест
    print("📝 Генерация манифеста...")
    manifest_path = generate_manifest(str(output_dir), stats, {})
    print(f"✅ Манифест создан: {manifest_path}")

    # 7. Валидация
    print("🔍 Валидация...")
    validate_zarr(str(zarr_path), manifest_path)
    print(f"✅ Сплит {split} успешно подготовлен!")


@cli.command()
@click.option("--config", default="configs/data/era5_28ch_full.yaml", help="Путь к YAML-конфигу")
def statistics(config):
    """Пересчитать статистики для train-сплита."""
    print("📊 Пересчёт статистик...")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["storage"]["output_dir"])
    zarr_path = output_dir / "train.zarr"

    if not zarr_path.exists():
        print(f"❌ Zarr не найден: {zarr_path}. Сначала запустите prepare --split train")
        sys.exit(1)

    ds = xr.open_zarr(zarr_path)
    stats = compute_train_statistics(ds)

    stats_path = output_dir / "statistics.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Статистики сохранены в {stats_path}")


@cli.command()
@click.option("--config", default="configs/data/era5_28ch_full.yaml", help="Путь к YAML-конфигу")
def validate(config):
    """Валидация готового датасета."""
    print("🔍 Валидация датасета...")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["storage"]["output_dir"])
    manifest_path = cfg["storage"]["manifest_path"]

    for split in ["train", "val", "test"]:
        zarr_path = output_dir / f"{split}.zarr"
        if not zarr_path.exists():
            print(f"⚠️ Пропуск {split}: {zarr_path} не найден")
            continue

        print(f"Проверка {split}...")
        validate_zarr(str(zarr_path), manifest_path)
        print(f"✅ {split} валиден")

    print("✅ Валидация завершена.")


if __name__ == "__main__":
    cli()
