import json
import hashlib
import xarray as xr
import numpy as np


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

    # Восстанавливаем "чистый" словарь без хеша
    manifest_copy = json.loads(json.dumps(manifest))  # Глубокая копия
    del manifest_copy["integrity"]["manifest_sha256"]
    canonical_json = json.dumps(manifest_copy, indent=2, sort_keys=True)
    computed_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    assert stored_hash == computed_hash, "manifest_sha256 не совпадает с хешем содержимого"

    return True