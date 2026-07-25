import xarray as xr
import numpy as np


def _to_python_serializable(val):
    """Конвертирует xarray/numpy в нативные Python типы (скаляр или список)."""
    if hasattr(val, 'values'):
        val = val.values
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()  # Скаляр для 1 элемента
        else:
            return val.tolist()  # Список для нескольких элементов
    return val


def compute_train_statistics(ds_train: xr.Dataset) -> dict:
    assert ds_train.chunks is not None, "Датасет должен быть чанкован"

    stats = {}
    for var in ds_train.data_vars:
        arr = ds_train[var]
        dims_to_reduce = [d for d in arr.dims if d != "channel"]

        # 1. Valid count
        valid_count_raw = arr.notnull().sum(dim=dims_to_reduce).compute()
        valid_count = _to_python_serializable(valid_count_raw)

        # 2. Mean & Std
        mean_raw = arr.mean(dim=dims_to_reduce, skipna=True).compute()
        mean_val = _to_python_serializable(mean_raw)

        std_raw = arr.std(dim=dims_to_reduce, skipna=True).compute()
        std_val = _to_python_serializable(std_raw)

        # 3. Percentiles
        flat_arr = arr.values.flatten()
        flat_arr = flat_arr[~np.isnan(flat_arr)]

        if len(flat_arr) > 0:
            p05 = float(np.percentile(flat_arr, 0.5))
            p995 = float(np.percentile(flat_arr, 99.5))
        else:
            p05, p995 = 0.0, 0.0

        stats[var] = {
            "mean": mean_val, "std": std_val,
            "p05": p05, "p995": p995, "valid_count": valid_count
        }
    return stats