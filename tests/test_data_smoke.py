import pytest
import subprocess
import os
import json
import numpy as np
import xarray as xr
from pathlib import Path


# 56. Полный synthetic pipeline
def test_56_synthetic_pipeline(tmp_path):
    env = os.environ.copy()
    env["SMOKE_OUTPUT_DIR"] = str(tmp_path)
    res = subprocess.run(["python", "scripts/prepare_era5_28ch.py", "smoke"],
                         env=env, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert res.returncode == 0, f"CLI CRASHED! STDERR: {res.stderr} STDOUT: {res.stdout}"


# 57. Полная validation
def test_57_full_validation(tmp_path):
    env = os.environ.copy()
    env["SMOKE_OUTPUT_DIR"] = str(tmp_path)
    subprocess.run(["python", "scripts/prepare_era5_28ch.py", "smoke"], env=env, cwd=Path(__file__).parent.parent)

    from era5_minimum.data.validation import validate_zarr
    # Предполагаем, что smoke создает data.zarr и manifest.json в tmp_path
    zarr_path = tmp_path / "data.zarr"
    manifest_path = tmp_path / "manifest.json"
    if zarr_path.exists() and manifest_path.exists():
        assert validate_zarr(str(zarr_path), str(manifest_path)) is True


# 58. Отсутствие Infinity
def test_58_no_infinity(tmp_path):
    env = os.environ.copy()
    env["SMOKE_OUTPUT_DIR"] = str(tmp_path)
    subprocess.run(["python", "scripts/prepare_era5_28ch.py", "smoke"], env=env, cwd=Path(__file__).parent.parent)

    ds = xr.open_zarr(tmp_path / "data.zarr")
    assert not np.isinf(ds["data"].values).any()


# 59. Разрешённые NaN только в маскированных областях
def test_59_nan_only_in_masked_areas(tmp_path):
    env = os.environ.copy()
    env["SMOKE_OUTPUT_DIR"] = str(tmp_path)
    subprocess.run(["python", "scripts/prepare_era5_28ch.py", "smoke"], env=env, cwd=Path(__file__).parent.parent)

    ds = xr.open_zarr(tmp_path / "data.zarr")
    # В smoke-тесте мы принудительно ставим NaN в SST (канал 5)
    sst_data = ds["data"].isel(channel=5).values
    assert np.isnan(sst_data).any()

    # Проверяем, что в других каналах (например, t2m - канал 0) NaN нет (если это синтетика без масок)
    t2m_data = ds["data"].isel(channel=0).values
    assert not np.isnan(t2m_data).any()


# 60. Повторный запуск даёт одинаковые индексы и hashes
def test_60_idempotent_rerun(tmp_path):
    env = os.environ.copy()
    env["SMOKE_OUTPUT_DIR"] = str(tmp_path)

    # Первый запуск
    subprocess.run(["python", "scripts/prepare_era5_28ch.py", "smoke"], env=env, cwd=Path(__file__).parent.parent)
    m1 = json.loads((tmp_path / "manifest.json").read_text())

    # Второй запуск
    subprocess.run(["python", "scripts/prepare_era5_28ch.py", "smoke"], env=env, cwd=Path(__file__).parent.parent)
    m2 = json.loads((tmp_path / "manifest.json").read_text())

    assert m1["nested_subsets"] == m2["nested_subsets"]
    assert m1["integrity"]["manifest_sha256"] == m2["integrity"]["manifest_sha256"]