"""
Реализует разделы 7 и 8 PCA_BASELINE.md — загрузку одного NPZ или
директории с несколькими NPZ (текущий формат ERA5-loader'а проекта:
ключи data/sst_mask/timestamps/latitude/longitude/channel_names/units),
их согласованную валидацию и строгий temporal split.

Важно: этот файл НЕ реализует сам ERA5-loader (его трогать запрещено
по условиям задачи) — он лишь читает уже готовые NPZ этого формата и
объединяет несколько дневных файлов в один датасет для baseline'а.

Проверяет несовместимость дневных файлов явно и с указанием причины
(разный порядок каналов, разные units, разные координаты, разные
sst_mask) — молчаливое объединение запрещено требованиями задачи (п.7).

Также реализует temporal split (train/val/test строго по времени, без
перемешивания) — критично для защиты от data leakage (п.8).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_REQUIRED_KEYS = ("data", "sst_mask", "timestamps", "latitude", "longitude", "channel_names", "units")


@dataclass
class ERA5NPZDataset:
    """Объединённый, провалидированный набор данных, готовый к patch-разбиению."""

    data: np.ndarray  # [time, channel, lat, lon]
    sst_mask: np.ndarray  # булев/0-1 массив, True/1 = валидная (океан) точка sst
    timestamps: np.ndarray  # [time]
    latitude: np.ndarray  # [lat]
    longitude: np.ndarray  # [lon]
    channel_names: list[str]
    units: dict[str, str]

    @property
    def n_timestamps(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]


@dataclass
class TemporalSplit:
    train: ERA5NPZDataset
    validation: ERA5NPZDataset
    test: ERA5NPZDataset


def _load_single_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        missing = [k for k in _REQUIRED_KEYS if k not in npz.files]
        if missing:
            raise ValueError(f"Файл {path}: отсутствуют обязательные ключи {missing}")
        payload = {k: npz[k] for k in _REQUIRED_KEYS}
    # channel_names/units часто сохраняются как 0-d object array — приводим к python-типам.
    channel_names = payload["channel_names"]
    if isinstance(channel_names, np.ndarray):
        channel_names = channel_names.tolist()
    payload["channel_names"] = list(channel_names)

    units = payload["units"]
    if isinstance(units, np.ndarray):
        units = units.item() if units.shape == () else units.tolist()
    if isinstance(units, list):
        units = dict(zip(payload["channel_names"], units))
    payload["units"] = dict(units)

    if payload["data"].ndim != 4:
        raise ValueError(
            f"Файл {path}: ожидалась форма data [time, channel, lat, lon], "
            f"получено ndim={payload['data'].ndim}"
        )
    return payload


def _check_consistency(reference: dict, candidate: dict, ref_path: Path, cand_path: Path) -> None:
    """Явные проверки совместимости дневных файлов (п.7 PCA_BASELINE.md).
    При несовпадении бросает ValueError с указанием конкретного файла и причины.
    """
    if candidate["channel_names"] != reference["channel_names"]:
        raise ValueError(
            f"Несовместимый порядок каналов в {cand_path} относительно {ref_path}: "
            f"{candidate['channel_names']} != {reference['channel_names']}"
        )
    if candidate["units"] != reference["units"]:
        raise ValueError(f"Несовпадающие units в {cand_path} относительно {ref_path}")
    if not np.array_equal(candidate["latitude"], reference["latitude"]):
        raise ValueError(f"Несовпадающая latitude в {cand_path} относительно {ref_path}")
    if not np.array_equal(candidate["longitude"], reference["longitude"]):
        raise ValueError(f"Несовпадающая longitude в {cand_path} относительно {ref_path}")
    if candidate["data"].shape[1:] != reference["data"].shape[1:]:
        raise ValueError(
            f"Несовпадающая форма [channel, lat, lon] в {cand_path}: "
            f"{candidate['data'].shape[1:]} != {reference['data'].shape[1:]}"
        )
    if candidate["sst_mask"].shape[1:] != reference["sst_mask"].shape[1:]:
        raise ValueError(f"Несовместимая форма sst_mask в {cand_path} относительно {ref_path}")


def load_era5_npz_dataset(input_path: str | Path) -> ERA5NPZDataset:
    """Загружает один NPZ либо директорию с несколькими NPZ, валидирует их
    совместимость, сортирует по timestamps и проверяет отсутствие дублей
    (пп.7.3-7.5, 7.9 PCA_BASELINE.md).
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"В директории {input_path} не найдено ни одного .npz файла")
    else:
        files = [input_path]

    payloads = [_load_single_npz(f) for f in files]
    reference = payloads[0]
    for path, payload in zip(files[1:], payloads[1:]):
        _check_consistency(reference, payload, files[0], path)

    data = np.concatenate([p["data"] for p in payloads], axis=0)
    sst_mask = np.concatenate([p["sst_mask"] for p in payloads], axis=0)
    timestamps = np.concatenate([np.asarray(p["timestamps"]).reshape(-1) for p in payloads], axis=0)

    order = np.argsort(timestamps, kind="stable")
    if not np.array_equal(order, np.arange(len(timestamps))):
        data, sst_mask, timestamps = data[order], sst_mask[order], timestamps[order]

    unique_ts, counts = np.unique(timestamps, return_counts=True)
    duplicates = unique_ts[counts > 1]
    if duplicates.size > 0:
        raise ValueError(f"Обнаружены дублирующиеся timestamps: {duplicates.tolist()}")

    return ERA5NPZDataset(
        data=data,
        sst_mask=sst_mask,
        timestamps=timestamps,
        latitude=np.asarray(reference["latitude"]),
        longitude=np.asarray(reference["longitude"]),
        channel_names=list(reference["channel_names"]),
        units=dict(reference["units"]),
    )


def temporal_split(
    dataset: ERA5NPZDataset,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> TemporalSplit:
    """Строгий temporal split (ранние -> train, следующие -> val, последние
    -> test) БЕЗ перемешивания timestamps (п.8 PCA_BASELINE.md). Датасет,
    возвращённый load_era5_npz_dataset, уже отсортирован по времени, поэтому
    здесь достаточно последовательных срезов.
    """
    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"train_fraction + validation_fraction + test_fraction должны давать 1.0, получено {total}"
        )
    n = dataset.n_timestamps
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * validation_fraction))
    n_train = max(1, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError("Некорректные fractions: test-часть получилась отрицательной")

    def _slice(start: int, end: int) -> ERA5NPZDataset:
        return ERA5NPZDataset(
            data=dataset.data[start:end],
            sst_mask=dataset.sst_mask[start:end],
            timestamps=dataset.timestamps[start:end],
            latitude=dataset.latitude,
            longitude=dataset.longitude,
            channel_names=dataset.channel_names,
            units=dataset.units,
        )

    train = _slice(0, n_train)
    validation = _slice(n_train, n_train + n_val)
    test = _slice(n_train + n_val, n)
    if train.n_timestamps == 0:
        raise ValueError("Train-часть пуста после split — проверьте train_fraction и размер датасета")
    return TemporalSplit(train=train, validation=validation, test=test)


def generate_synthetic_dataset(
    n_timestamps: int = 40,
    height: int = 37,
    width: int = 72,
    channel_names: list[str] | None = None,
    seed: int = 0,
) -> ERA5NPZDataset:
    """Генерирует небольшой synthetic ERA5-подобный датасет для --smoke-test
    (раздел 18 PCA_BASELINE.md / п.18-19 TASKA_mat_cod.txt).

    Требования к smoke-датасету:
        - работает без интернета и без реальных данных (data/, outputs/);
        - несколько каналов (по умолчанию канонический набор из 8);
        - содержит NaN / masked channel (sst с NaN над "сушей");
        - размер намеренно НЕ кратен стандартному patch_size 8x8
          (высота 37, ширина 72), чтобы заодно проверялся padding.

    Не предназначена для использования как источник "реальных" метрик —
    is_demo для всех артефактов, построенных на этих данных, должен быть True.
    """
    if channel_names is None:
        channel_names = ["u10", "v10", "t2m", "msl", "sst", "tcc", "tcwv", "tp1h"]
    rng = np.random.default_rng(seed)
    n_channels = len(channel_names)

    data = rng.normal(loc=0.0, scale=1.0, size=(n_timestamps, n_channels, height, width)).astype(np.float32)
    scale_by_channel = {"t2m": 15.0, "msl": 500.0, "sst": 5.0, "tcwv": 10.0, "tp1h": 0.002}
    offset_by_channel = {"t2m": 288.0, "msl": 101325.0, "sst": 290.0, "tcc": 0.5, "tp1h": 0.001}
    for c, name in enumerate(channel_names):
        data[:, c] = data[:, c] * scale_by_channel.get(name, 1.0) + offset_by_channel.get(name, 0.0)
    if "tcc" in channel_names:
        c = channel_names.index("tcc")
        data[:, c] = np.clip(data[:, c], 0.0, 1.0)
    if "tp1h" in channel_names:
        c = channel_names.index("tp1h")
        data[:, c] = np.clip(data[:, c], 0.0, None)

    land_fraction = 0.35
    sst_mask_2d = rng.uniform(size=(height, width)) > land_fraction
    sst_mask = np.broadcast_to(sst_mask_2d, (n_timestamps, height, width)).copy()

    if "sst" in channel_names:
        c = channel_names.index("sst")
        data[:, c][:, ~sst_mask_2d] = np.nan

    timestamps = np.array(
        [np.datetime64("2024-01-01T00:00:00") + np.timedelta64(h, "h") for h in range(n_timestamps)]
    )
    latitude = np.linspace(-90, 90, height, dtype=np.float32)
    longitude = np.linspace(0, 360, width, endpoint=False, dtype=np.float32)
    default_units = ["m s**-1", "m s**-1", "K", "Pa", "K", "0-1", "kg m**-2", "m"]
    units = {
        name: (default_units[i] if i < len(default_units) else "1")
        for i, name in enumerate(channel_names)
    }

    return ERA5NPZDataset(
        data=data,
        sst_mask=sst_mask,
        timestamps=timestamps,
        latitude=latitude,
        longitude=longitude,
        channel_names=channel_names,
        units=units,
    )