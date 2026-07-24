"""
Реализует шаги пайплайна №№ 4-5 и 8-9 из PCA_BASELINE.md:
    - расчёт zero-padding для сетки, не кратной размеру патча (361x720);
    - разбиение карты `[batch, channel, height, width]` на
      неперекрывающиеся патчи `[n_patches, patch_features]`;
    - обратную сборку карты из патчей и удаление padding.

Это единственное место в проекте, где определён порядок flatten/unflatten
патча в вектор признаков. Любой другой модуль (patch_pca.py, скрипты)
обязан использовать функции отсюда, а не реализовывать разбиение заново.

Файл не знает ничего про PCA, нормализацию или метеорологические каналы —
только про геометрию патчей. Это сознательное разделение ответственности:
патчинг должен быть переиспользуем и для будущего ConvAE-бaзлайна.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchGridInfo:
    """Вся геометрия, необходимая для round-trip extract -> reconstruct.

    Хранит и исходную, и padded-форму, поэтому reconstruct_from_patches
    не нуждается в дополнительных аргументах, кроме этого объекта.
    """

    batch_size: int
    channel_count: int
    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    patch_height: int
    patch_width: int
    pad_bottom: int
    pad_right: int
    n_patches_h: int
    n_patches_w: int

    @property
    def patch_feature_count(self) -> int:
        return self.channel_count * self.patch_height * self.patch_width

    @property
    def n_patches_per_sample(self) -> int:
        return self.n_patches_h * self.n_patches_w

    @property
    def n_patches_total(self) -> int:
        return self.batch_size * self.n_patches_per_sample


def compute_padding(size: int, patch_size: int) -> int:
    """Возвращает величину padding, необходимую, чтобы `size` стал кратен
    `patch_size`. Реализует правило п.4.2 PCA_BASELINE.md: padding
    добавляется только справа/снизу, строки/столбцы никогда не отбрасываются.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size должен быть положительным, получено {patch_size}")
    remainder = size % patch_size
    return 0 if remainder == 0 else patch_size - remainder


def patch_grid_dims(height: int, width: int, patch_height: int, patch_width: int) -> tuple[int, int]:
    """Возвращает (n_patches_h, n_patches_w) без выполнения самого padding/
    разбиения — удобно, когда нужно заранее посчитать общее число патчей
    (например, для n_train_patches в CLI), не материализуя массив."""
    pad_bottom = compute_padding(height, patch_height)
    pad_right = compute_padding(width, patch_width)
    return (height + pad_bottom) // patch_height, (width + pad_right) // patch_width


def _pad_map(data: np.ndarray, patch_height: int, patch_width: int) -> tuple[np.ndarray, int, int]:
    """Внутренняя функция: добавляет zero-padding справа/снизу к карте формы
    [batch, channel, height, width]. Padding явно фиксирован как zero padding
    (см. п.6 PCA_BASELINE.md), выполняется уже ПОСЛЕ нормализации — вызывающий
    код (patch_pca.py) отвечает за порядок вызовов.
    """
    _, _, height, width = data.shape
    pad_bottom = compute_padding(height, patch_height)
    pad_right = compute_padding(width, patch_width)
    if pad_bottom == 0 and pad_right == 0:
        return data, 0, 0
    padded = np.pad(
        data,
        pad_width=((0, 0), (0, 0), (0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=0.0,
    )
    return padded, pad_bottom, pad_right


def extract_patches(
    data: np.ndarray, patch_height: int, patch_width: int
) -> tuple[np.ndarray, PatchGridInfo]:
    """Разбивает карту на неперекрывающиеся патчи.

    Args:
        data: массив формы [batch, channel, height, width]. `batch` может
            быть, например, числом timestamp'ов.
        patch_height, patch_width: размер патча.

    Returns:
        patches: массив формы [n_patches_total, patch_feature_count], где
            patch_feature_count = channel_count * patch_height * patch_width.
            Порядок патчей: сначала по batch, затем по строкам патч-сетки,
            затем по столбцам (row-major). Порядок flatten внутри патча:
            channel-major, затем строки патча, затем столбцы
            (совпадает с C-order reshape(channel, ph, pw) -> flatten).
        grid: PatchGridInfo, необходимый для reconstruct_from_patches.

    Raises:
        ValueError: если data не 4-мерный массив или patch_size некорректен.
    """
    if data.ndim != 4:
        raise ValueError(
            f"Ожидалась форма [batch, channel, height, width], получено ndim={data.ndim}"
        )
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch_height и patch_width должны быть положительными")

    batch_size, channel_count, height, width = data.shape
    padded, pad_bottom, pad_right = _pad_map(data, patch_height, patch_width)
    padded_height, padded_width = padded.shape[2], padded.shape[3]
    n_patches_h = padded_height // patch_height
    n_patches_w = padded_width // patch_width

    # [batch, channel, n_patches_h, patch_height, n_patches_w, patch_width]
    reshaped = padded.reshape(
        batch_size, channel_count, n_patches_h, patch_height, n_patches_w, patch_width
    )
    # -> [batch, n_patches_h, n_patches_w, channel, patch_height, patch_width]
    transposed = reshaped.transpose(0, 2, 4, 1, 3, 5)
    patches = transposed.reshape(
        batch_size * n_patches_h * n_patches_w,
        channel_count * patch_height * patch_width,
    )

    grid = PatchGridInfo(
        batch_size=batch_size,
        channel_count=channel_count,
        original_height=height,
        original_width=width,
        padded_height=padded_height,
        padded_width=padded_width,
        patch_height=patch_height,
        patch_width=patch_width,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        n_patches_h=n_patches_h,
        n_patches_w=n_patches_w,
    )
    return patches, grid


def reconstruct_from_patches(patches: np.ndarray, grid: PatchGridInfo) -> np.ndarray:
    """Обратная операция к extract_patches: собирает карту из патчей и
    обрезает padding так, чтобы результат совпадал по форме со входом,
    поданным в extract_patches (см. требование п.4.2:
    reconstructed.shape == original.shape).
    """
    expected_n_patches = grid.n_patches_total
    expected_features = grid.patch_feature_count
    if patches.shape != (expected_n_patches, expected_features):
        raise ValueError(
            "Форма patches не совпадает с PatchGridInfo: "
            f"ожидалось {(expected_n_patches, expected_features)}, получено {patches.shape}"
        )

    reshaped = patches.reshape(
        grid.batch_size,
        grid.n_patches_h,
        grid.n_patches_w,
        grid.channel_count,
        grid.patch_height,
        grid.patch_width,
    )
    # -> [batch, channel, n_patches_h, patch_height, n_patches_w, patch_width]
    transposed = reshaped.transpose(0, 3, 1, 4, 2, 5)
    padded = transposed.reshape(
        grid.batch_size, grid.channel_count, grid.padded_height, grid.padded_width
    )

    if grid.pad_bottom == 0 and grid.pad_right == 0:
        return padded

    height_end = grid.padded_height - grid.pad_bottom
    width_end = grid.padded_width - grid.pad_right
    return padded[:, :, :height_end, :width_end]