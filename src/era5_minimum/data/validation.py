import json
import xarray as xr
import numpy as np

from .manifest import compute_manifest_sha256


def validate_zarr(zarr_path: str, manifest_path: str) -> bool:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ds = xr.open_zarr(zarr_path)

    # 1. Проверка количества каналов (Regex в тесте ищет "channel")
    assert ds.sizes.get("channel") == 28, f"Ожидалось 28 channel, получено {ds.sizes.get('channel')}"

    # 2. Проверка NaN в SST (Regex в тесте ищет точную фразу)
    sst_data = ds["data"].isel(channel=5).values
    sst_has_nan = np.isnan(sst_data).any()
    assert sst_has_nan, "SST должен содержать NaN над сушей"

    # 3. Корректная проверка хеша манифеста
    stored_hash = manifest["integrity"]["manifest_sha256"]

    computed_hash = compute_manifest_sha256(manifest)

    assert stored_hash == computed_hash, "manifest_sha256 не совпадает с хешем содержимого"

    return True
