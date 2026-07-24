"""
Реализует раздел 16 PCA_BASELINE.md: расчёт MSE/RMSE/MAE после
денормализации, отдельно overall и per-channel, с обязательным
исключением padding и невалидных точек (суша для sst, NaN для прочих
каналов) из расчёта — см. п.4.2 ("Padding не должен участвовать в
расчёте MSE, RMSE, MAE") и п.6.2 ("не включать сушу в MSE, RMSE и MAE").

Термин NRMSE намеренно не используется (см. п.16: "Не использовать
термин NRMSE, если его формула отдельно не определена") — только MSE,
RMSE, MAE.

Этот модуль ничего не знает о PCA и о том, как получена reconstruction —
он работает с любыми двумя массивами одинаковой формы плюс маской.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsResult:
    """Итог расчёта метрик: overall + per-channel."""

    overall: dict[str, float]
    per_channel: dict[str, dict[str, float]]
    valid_point_count: int

    def to_json_dict(self) -> dict:
        return {
            "overall": self.overall,
            "per_channel": self.per_channel,
            "valid_point_count": self.valid_point_count,
        }

    def assert_finite(self) -> None:
        """Проверка п.19.30 ('конечность всех метрик'): бросает исключение,
        если хоть одна метрика NaN/Inf."""
        for name, value in self.overall.items():
            if not np.isfinite(value):
                raise ValueError(f"Метрика overall.{name} не конечна: {value}")
        for channel, values in self.per_channel.items():
            for name, value in values.items():
                if not np.isfinite(value):
                    raise ValueError(f"Метрика per_channel[{channel}].{name} не конечна: {value}")


def _channel_metrics(original: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        raise ValueError(
            "Нет ни одной валидной точки для расчёта метрик "
            "(маска полностью пустая — проверьте padding/SST mask)"
        )
    diff = (original[mask] - reconstruction[mask]).astype(np.float64)
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def compute_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
    channel_names: list[str],
    valid_mask: np.ndarray | None = None,
) -> MetricsResult:
    """Считает overall и per-channel MSE/RMSE/MAE.

    Args:
        original, reconstruction: массивы формы [time, channel, height, width]
            в ФИЗИЧЕСКИХ единицах (после денормализации), уже БЕЗ padding
            (т.е. форма совпадает с исходной картой до patch-разбиения).
        channel_names: имена каналов по оси 1, той же длины, что и
            original.shape[1].
        valid_mask: булев массив. Допустимые формы:
              - совпадает с original.shape (например, SST-маска суши,
                своя на каждый timestamp/канал);
              - [time, height, width] — общая для всех каналов маска
                (например, если строится по padding или общей validity).
            True = точка учитывается в метриках. Если None — валидны все
            точки (используется, например, для каналов без NaN/масок).

    Returns:
        MetricsResult с overall и per_channel метриками, а также числом
        валидных точек, фактически использованных в расчёте.
    """
    if original.shape != reconstruction.shape:
        raise ValueError(
            f"original.shape {original.shape} != reconstruction.shape {reconstruction.shape}"
        )
    if original.shape[1] != len(channel_names):
        raise ValueError(
            f"Число каналов в данных ({original.shape[1]}) не совпадает с "
            f"channel_names ({len(channel_names)})"
        )

    finite_mask = np.isfinite(original) & np.isfinite(reconstruction)
    if valid_mask is not None:
        if valid_mask.shape == original.shape:
            full_mask = finite_mask & valid_mask
        elif valid_mask.shape == (original.shape[0], original.shape[2], original.shape[3]):
            full_mask = finite_mask & valid_mask[:, None, :, :]
        else:
            raise ValueError(
                f"Неожиданная форма valid_mask: {valid_mask.shape}, "
                f"ожидалось {original.shape} или {(original.shape[0], original.shape[2], original.shape[3])}"
            )
    else:
        full_mask = finite_mask

    per_channel: dict[str, dict[str, float]] = {}
    for c, name in enumerate(channel_names):
        per_channel[name] = _channel_metrics(original[:, c], reconstruction[:, c], full_mask[:, c])

    overall = _channel_metrics(original, reconstruction, full_mask)
    return MetricsResult(
        overall=overall,
        per_channel=per_channel,
        valid_point_count=int(full_mask.sum()),
    )