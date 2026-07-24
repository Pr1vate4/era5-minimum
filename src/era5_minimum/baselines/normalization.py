"""
Реализует раздел 6 и 9-10 PCA_BASELINE.md: потоковую (streaming),
численно устойчивую нормализацию метеорологических каналов,
рассчитываемую ИСКЛЮЧИТЕЛЬНО по train-части данных (защита от
data leakage, см. п.5.2 и п.8 спецификации).

Использует параллельный алгоритм Чана (обобщение Welford на батчи) —
объединение статистик (count, mean, M2) двух батчей без хранения всех
данных в памяти одновременно. Это то, что в спецификации названо
"потоковый Welford-подобный алгоритм".

Поддерживает:
    - игнорирование NaN (канал sst содержит NaN над сушей, п.10);
    - произвольную validity-маску для любого канала (на случай будущих
      пропусков в официальном датасете, не только sst);
    - безопасное поведение при std == 0 (фиксируется в metadata, а не
      молча заменяется без объяснений).

Сохраняет/загружает normalization.json в формате, заданном в п.9
PCA_BASELINE.md и разделе 8.2 контракта артефактов.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Значение std, которое используется, если реальный std канала равен 0
# (например, канал константный на train-выборке). Явно фиксируется в
# normalization.json через флаг used_safe_std, чтобы это не выглядело как
# "настоящая" статистика.
_SAFE_STD_VALUE = 1.0


@dataclass
class NormalizationStats:
    """Результат fit() ChannelNormalizer — то, что сохраняется в
    normalization.json согласно контракту (раздел 8.2 PCA_BASELINE.md).
    """

    channel_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    valid_count: np.ndarray
    used_safe_std: list[bool]
    source_split: str = "train"

    def to_json_dict(self) -> dict:
        return {
            "channel_names": list(self.channel_names),
            "mean": np.asarray(self.mean).tolist(),
            "std": np.asarray(self.std).tolist(),
            "valid_count": np.asarray(self.valid_count).astype(int).tolist(),
            "used_safe_std": list(self.used_safe_std),
            "source_split": self.source_split,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "NormalizationStats":
        return cls(
            channel_names=list(payload["channel_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            valid_count=np.asarray(payload["valid_count"], dtype=np.int64),
            used_safe_std=list(payload.get("used_safe_std", [False] * len(payload["channel_names"]))),
            source_split=payload.get("source_split", "train"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_json_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_json_dict(json.load(fh))


def _combine(count_a, mean_a, m2_a, count_b, mean_b, m2_b):
    """Параллельное объединение статистик Уэлфорда (алгоритм Чана).
    Все аргументы — массивы per-channel формы [channel_count]."""
    total_count = count_a + count_b
    # Избегаем деления на 0 там, где обоих батчей ещё не было данных.
    safe_total = np.where(total_count > 0, total_count, 1)
    delta = mean_b - mean_a
    mean = mean_a + delta * (count_b / safe_total)
    m2 = m2_a + m2_b + delta**2 * (count_a * count_b / safe_total)
    return total_count, mean, m2


class ChannelNormalizer:
    """Потоковый расчёт mean/std по каналам без загрузки всего train-набора
    в память сразу — данные подаются порциями через update().
    """

    def __init__(self, channel_names: list[str]):
        self.channel_names = list(channel_names)
        n = len(self.channel_names)
        self._count = np.zeros(n, dtype=np.float64)
        self._mean = np.zeros(n, dtype=np.float64)
        self._m2 = np.zeros(n, dtype=np.float64)
        self._fitted = False

    def update(self, batch: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        """Добавляет очередную порцию данных.

        Args:
            batch: массив формы [time, channel, height, width] (или любой
                формы, где ось 1 — канал; остальные оси трактуются как
                выборочные точки).
            valid_mask: булев массив той же формы, что и batch, без оси
                канала свёрнутой отдельно — либо форма [time, height, width]
                (общая для всех каналов, например padding-маска), либо
                полная форма batch.shape (например SST-маска по суше/морю).
                True = точка валидна и учитывается в статистике.
                Если None, валидны все точки, кроме NaN (NaN всегда
                исключается автоматически).
        """
        if batch.ndim < 2:
            raise ValueError("batch должен иметь минимум оси [*, channel, ...]")
        n_channels = batch.shape[1]
        if n_channels != len(self.channel_names):
            raise ValueError(
                f"Число каналов в batch ({n_channels}) не совпадает с "
                f"channel_names ({len(self.channel_names)})"
            )

        nan_mask = ~np.isnan(batch)
        for c in range(n_channels):
            channel_data = batch[:, c, ...]
            channel_valid = nan_mask[:, c, ...]
            if valid_mask is not None:
                if valid_mask.shape == batch.shape:
                    channel_valid = channel_valid & valid_mask[:, c, ...]
                else:
                    channel_valid = channel_valid & valid_mask
            values = channel_data[channel_valid]
            if values.size == 0:
                continue
            batch_count = float(values.size)
            batch_mean = float(values.mean())
            batch_m2 = float(((values - batch_mean) ** 2).sum())
            total_count, mean, m2 = _combine(
                self._count[c], self._mean[c], self._m2[c], batch_count, batch_mean, batch_m2
            )
            self._count[c] = total_count
            self._mean[c] = mean
            self._m2[c] = m2
        self._fitted = True

    def finalize(self, source_split: str = "train") -> NormalizationStats:
        """Завершает потоковый расчёт и возвращает NormalizationStats.

        При std == 0 (или valid_count == 0) подставляет безопасное значение
        _SAFE_STD_VALUE и фиксирует это в used_safe_std, как того требует
        п.9 PCA_BASELINE.md ("при std=0 использовать безопасное значение и
        зафиксировать это в metadata").
        """
        if not self._fitted:
            raise RuntimeError("ChannelNormalizer.update() ни разу не вызывался")
        variance = np.where(self._count > 1, self._m2 / np.maximum(self._count, 1), 0.0)
        std = np.sqrt(variance)
        used_safe_std = (std == 0) | (self._count == 0)
        safe_std = np.where(used_safe_std, _SAFE_STD_VALUE, std)
        return NormalizationStats(
            channel_names=self.channel_names,
            mean=self._mean.copy(),
            std=safe_std,
            valid_count=self._count.astype(np.int64),
            used_safe_std=used_safe_std.tolist(),
            source_split=source_split,
        )


def normalize(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Применяет train-only нормализацию (data - mean) / std к массиву формы
    [..., channel, ...]. Ось канала должна быть осью 1, как и в update().
    """
    mean = stats.mean.reshape(1, -1, *([1] * (data.ndim - 2)))
    std = stats.std.reshape(1, -1, *([1] * (data.ndim - 2)))
    return (data - mean) / std


def denormalize(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Обратная операция к normalize(): возвращает данные в физические
    единицы (см. шаг 10 общего пайплайна в PCA_BASELINE.md)."""
    mean = stats.mean.reshape(1, -1, *([1] * (data.ndim - 2)))
    std = stats.std.reshape(1, -1, *([1] * (data.ndim - 2)))
    return data * std + mean