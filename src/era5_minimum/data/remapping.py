import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix


def _cell_bounds(centers: np.ndarray) -> np.ndarray:
    """
    Вычисляет границы ячеек из их центров.
    Для N центров возвращает N+1 границу.
    """
    bounds = np.empty(len(centers) + 1)
    bounds[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    # Крайние границы extrapolate от первых/последних двух центров
    bounds[0] = centers[0] - (bounds[1] - centers[0])
    bounds[-1] = centers[-1] + (centers[-1] - bounds[-2])
    return bounds


def _build_conservative_weights(
    lat_in: np.ndarray,
    lon_in: np.ndarray,
    lat_out: np.ndarray,
    lon_out: np.ndarray,
) -> csr_matrix:
    """
    Строит sparse weight matrix для first-order conservative remapping
    между регулярными широтно-долготными сетками.

    Учитывает:
    - Сферическую геометрию (вес пропорционален cos(lat))
    - Периодичность longitude (wrap-around 360°)

    Возвращает матрицу W формы (n_out, n_in), где:
    - n_out = len(lat_out) * len(lon_out)
    - n_in  = len(lat_in) * len(lon_in)
    - W[j, i] = доля площади target ячейки j, покрытая source ячейкой i
    """
    # 1. Границы ячеек
    lat_in_b = _cell_bounds(lat_in)
    lon_in_b = _cell_bounds(lon_in)
    lat_out_b = _cell_bounds(lat_out)
    lon_out_b = _cell_bounds(lon_out)

    # Ограничиваем широтные границы полюсами
    lat_in_b = np.clip(lat_in_b, -90.0, 90.0)
    lat_out_b = np.clip(lat_out_b, -90.0, 90.0)

    n_lat_in = len(lat_in)
    n_lon_in = len(lon_in)
    n_lat_out = len(lat_out)
    n_lon_out = len(lon_out)

    n_in = n_lat_in * n_lon_in
    n_out = n_lat_out * n_lon_out

    # 2. Предвычисляем sin(lat) для площадей (интеграл cos(lat) dlat = sin(lat))
    sin_lat_in = np.sin(np.deg2rad(lat_in_b))
    sin_lat_out = np.sin(np.deg2rad(lat_out_b))

    # 3. Строим веса
    rows = []
    cols = []
    weights = []

    for j_lat in range(n_lat_out):
        # Широтное перекрытие: площадь полосы = |sin(lat_top) - sin(lat_bot)|
        s_out = abs(sin_lat_out[j_lat + 1] - sin_lat_out[j_lat])
        if s_out < 1e-15:
            continue

        # Находим перекрывающиеся широтные ячейки source
        lat_lo = lat_out_b[j_lat]
        lat_hi = lat_out_b[j_lat + 1]

        for i_lat in range(n_lat_in):
            # Пересечение широтных интервалов
            overlap_lat_lo = max(lat_lo, lat_in_b[i_lat])
            overlap_lat_hi = min(lat_hi, lat_in_b[i_lat + 1])

            if overlap_lat_hi <= overlap_lat_lo:
                continue

            # Площадь широтного перекрытия
            s_overlap = abs(
                np.sin(np.deg2rad(overlap_lat_hi))
                - np.sin(np.deg2rad(overlap_lat_lo))
            )
            lat_weight = s_overlap / s_out

            for j_lon in range(n_lon_out):
                lon_lo = lon_out_b[j_lon]
                lon_hi = lon_out_b[j_lon + 1]
                d_lon_out = lon_hi - lon_lo

                if d_lon_out < 1e-15:
                    continue

                for i_lon in range(n_lon_in):
                    # Долготное перекрытие с учётом периодичности
                    in_lo = lon_in_b[i_lon]
                    in_hi = lon_in_b[i_lon + 1]

                    # Нормализуем в [0, 360)
                    overlap_lon = _lon_overlap(lon_lo, lon_hi, in_lo, in_hi)

                    if overlap_lon < 1e-15:
                        continue

                    lon_weight = overlap_lon / d_lon_out
                    w = lat_weight * lon_weight

                    if w > 1e-12:
                        j = j_lat * n_lon_out + j_lon
                        i = i_lat * n_lon_in + i_lon
                        rows.append(j)
                        cols.append(i)
                        weights.append(w)

    W = csr_matrix((weights, (rows, cols)), shape=(n_out, n_in))

    # 4. Нормализация: сумма весов по строкам должна быть 1
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums < 1e-15] = 1.0  # избегаем деления на ноль
    W = W.multiply(1.0 / row_sums[:, np.newaxis])

    return W


def _lon_overlap(out_lo: float, out_hi: float, in_lo: float, in_hi: float) -> float:
    """
    Вычисляет длину перекрытия двух долготных интервалов
    с учётом периодичности 360°.
    """
    # Нормализуем все координаты в [0, 360)
    out_lo_n = out_lo % 360.0
    out_hi_n = out_hi % 360.0
    in_lo_n = in_lo % 360.0
    in_hi_n = in_hi % 360.0

    # Обрабатываем случай, когда интервал пересекает 0°/360°
    # Разбиваем каждый интервал на под-интервалы в [0, 360)
    out_segments = _split_periodic(out_lo_n, out_hi_n, out_hi - out_lo)
    in_segments = _split_periodic(in_lo_n, in_hi_n, in_hi - in_lo)

    total_overlap = 0.0
    for o_lo, o_hi in out_segments:
        for i_lo, i_hi in in_segments:
            overlap = max(0.0, min(o_hi, i_hi) - max(o_lo, i_lo))
            total_overlap += overlap

    return total_overlap


def _split_periodic(lo: float, hi: float, width: float) -> list:
    """
    Разбивает периодический интервал на под-интервалы в [0, 360).
    """
    if width >= 360.0:
        return [(0.0, 360.0)]

    lo = lo % 360.0
    hi = (lo + width) % 360.0

    if hi > lo:
        return [(lo, hi)]
    else:
        # Пересекает 0°
        return [(lo, 360.0), (0.0, hi)]


def conservative_remap(ds_in: xr.Dataset, ds_out_grid: xr.Dataset) -> xr.Dataset:
    """
    First-order conservative remapping с учётом периодичности longitude.
    Реализация через scipy.sparse (без зависимости от xesmf/ESMF).

    Args:
        ds_in: Исходный xarray.Dataset с координатами latitude, longitude.
        ds_out_grid: Целевая сетка (xr.Dataset с latitude, longitude).

    Returns:
        xr.Dataset на целевой сетке.
    """
    lat_in = ds_in["latitude"].values
    lon_in = ds_in["longitude"].values
    lat_out = ds_out_grid["latitude"].values
    lon_out = ds_out_grid["longitude"].values

    # Строим weight matrix один раз
    W = _build_conservative_weights(lat_in, lon_in, lat_out, lon_out)

    n_lat_out = len(lat_out)
    n_lon_out = len(lon_out)
    n_lat_in = len(lat_in)
    n_lon_in = len(lon_in)

    remapped_vars = {}

    for var_name in ds_in.data_vars:
        da = ds_in[var_name]
        dims = da.dims

        if "latitude" not in dims or "longitude" not in dims:
            # Переменная без пространственных измерений — копируем как есть
            remapped_vars[var_name] = da
            continue

        data = da.values

        if "level" in dims:
            # Форма: (time, level, lat, lon) или (level, lat, lon)
            if "time" in dims:
                n_time = data.shape[0]
                n_level = data.shape[1]
                out_shape = (n_time, n_level, n_lat_out, n_lon_out)
                data_out = np.full(out_shape, np.nan, dtype=np.float32)

                for t in range(n_time):
                    for l in range(n_level):
                        flat_in = data[t, l].ravel()
                        # Обрабатываем NaN: заменяем на 0 для умножения,
                        # затем маскируем
                        mask = ~np.isnan(flat_in)
                        flat_in_safe = np.where(mask, flat_in, 0.0)

                        flat_out = W.dot(flat_in_safe)

                        # Маскируем ячейки, где все source были NaN
                        mask_out = W.dot(mask.astype(np.float64))
                        flat_out = np.where(mask_out > 0.5, flat_out, np.nan)

                        data_out[t, l] = flat_out.reshape(n_lat_out, n_lon_out)

                remapped_vars[var_name] = (
                    ["time", "level", "latitude", "longitude"],
                    data_out,
                )
            else:
                n_level = data.shape[0]
                out_shape = (n_level, n_lat_out, n_lon_out)
                data_out = np.full(out_shape, np.nan, dtype=np.float32)

                for l in range(n_level):
                    flat_in = data[l].ravel()
                    mask = ~np.isnan(flat_in)
                    flat_in_safe = np.where(mask, flat_in, 0.0)
                    flat_out = W.dot(flat_in_safe)
                    mask_out = W.dot(mask.astype(np.float64))
                    flat_out = np.where(mask_out > 0.5, flat_out, np.nan)
                    data_out[l] = flat_out.reshape(n_lat_out, n_lon_out)

                remapped_vars[var_name] = (
                    ["level", "latitude", "longitude"],
                    data_out,
                )
        else:
            # Surface variable: (time, lat, lon) или (lat, lon)
            if "time" in dims:
                n_time = data.shape[0]
                out_shape = (n_time, n_lat_out, n_lon_out)
                data_out = np.full(out_shape, np.nan, dtype=np.float32)

                for t in range(n_time):
                    flat_in = data[t].ravel()
                    mask = ~np.isnan(flat_in)
                    flat_in_safe = np.where(mask, flat_in, 0.0)
                    flat_out = W.dot(flat_in_safe)
                    mask_out = W.dot(mask.astype(np.float64))
                    flat_out = np.where(mask_out > 0.5, flat_out, np.nan)
                    data_out[t] = flat_out.reshape(n_lat_out, n_lon_out)

                remapped_vars[var_name] = (
                    ["time", "latitude", "longitude"],
                    data_out,
                )
            else:
                flat_in = data.ravel()
                mask = ~np.isnan(flat_in)
                flat_in_safe = np.where(mask, flat_in, 0.0)
                flat_out = W.dot(flat_in_safe)
                mask_out = W.dot(mask.astype(np.float64))
                flat_out = np.where(mask_out > 0.5, flat_out, np.nan)
                data_out = flat_out.reshape(n_lat_out, n_lon_out).astype(np.float32)
                remapped_vars[var_name] = (
                    ["latitude", "longitude"],
                    data_out,
                )

    # Собираем координаты
    coords = {
        "latitude": (["latitude"], lat_out),
        "longitude": (["longitude"], lon_out),
    }
    if "time" in ds_in.coords:
        coords["time"] = ds_in["time"]
    if "level" in ds_in.coords:
        coords["level"] = ds_in["level"]

    return xr.Dataset(remapped_vars, coords=coords)