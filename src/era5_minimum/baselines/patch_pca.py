"""
Центральный класс baseline'а — `PatchPCABaseline`. Реализует разделы
3-4 PCA_BASELINE.md: fit/encode/decode/reconstruct/save/load/
compression_info поверх `sklearn.decomposition.IncrementalPCA`.

Ключевая ответственность именно этого файла — БАТЧИНГ для
IncrementalPCA.partial_fit с буфером (п.4 PCA_BASELINE.md и п.4
TASKA_mat_cod.txt): гарантирует, что в partial_fit никогда не попадёт
порция меньше n_components, и ни один патч не будет молча отброшен.

Модуль опирается на patches.py (геометрия патчей) и не содержит
собственной логики разбиения на патчи — только математику PCA и
учёт коэффициентов сжатия (payload и end-to-end, раздел 12 спецификации).

НЕ реализует нейросети, backpropagation и т.п. — это чисто линейный
baseline, что явно требуется разделом 1 PCA_BASELINE.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from sklearn.decomposition import IncrementalPCA

from era5_minimum.baselines.patches import (
    PatchGridInfo,
    extract_patches,
    reconstruct_from_patches,
)

_MODEL_FILE = "pca_model.npz"


@dataclass
class CompressionInfo:
    """Результат compression_info() — раздел 3 (compression_info) и 12
    ("Коэффициент сжатия") PCA_BASELINE.md."""

    original_shape: tuple[int, ...]
    patch_height: int
    patch_width: int
    patch_feature_count: int
    n_components: int
    n_patches: int
    original_bytes: int
    coefficients_bytes: int
    model_bytes: int
    normalization_bytes: int
    metadata_bytes: int
    target_compression_ratio: float | None
    payload_compression_ratio: float
    end_to_end_compression_ratio: float

    def to_json_dict(self) -> dict:
        return {
            "original_shape": list(self.original_shape),
            "patch_height": self.patch_height,
            "patch_width": self.patch_width,
            "patch_feature_count": self.patch_feature_count,
            "n_components": self.n_components,
            "n_patches": self.n_patches,
            "original_bytes": self.original_bytes,
            "coefficients_bytes": self.coefficients_bytes,
            "model_bytes": self.model_bytes,
            "normalization_bytes": self.normalization_bytes,
            "metadata_bytes": self.metadata_bytes,
            "target_compression_ratio": self.target_compression_ratio,
            "payload_compression_ratio": round(self.payload_compression_ratio, 4),
            "end_to_end_compression_ratio": round(self.end_to_end_compression_ratio, 4),
        }


def resolve_n_components(
    patch_feature_count: int,
    n_train_patches: int,
    target_compression_ratio: float | None = None,
    n_components: int | None = None,
) -> int:
    """Реализует п.5/п.7.1 PCA_BASELINE.md: приоритет n_components над
    target_compression_ratio, минимум 1 компонента, и обязательные проверки
    (n_components <= patch_feature_count, <= число train-патчей).
    """
    if n_components is None and target_compression_ratio is None:
        raise ValueError("Нужно задать либо n_components, либо target_compression_ratio")

    if n_components is None:
        if target_compression_ratio <= 0:
            raise ValueError(f"target_compression_ratio должен быть положительным, получено {target_compression_ratio}")
        n_components = max(1, int(patch_feature_count // target_compression_ratio))

    if n_components <= 0:
        raise ValueError(f"n_components должен быть положительным, получено {n_components}")
    if n_components > patch_feature_count:
        raise ValueError(
            f"n_components ({n_components}) не может быть больше patch_feature_count ({patch_feature_count})"
        )
    if n_components > n_train_patches:
        raise ValueError(
            f"n_components ({n_components}) не может быть больше числа train-патчей ({n_train_patches})"
        )
    return n_components


class PatchPCABaseline:
    """Patch-based PCA baseline поверх sklearn.IncrementalPCA.

    Использование:
        model = PatchPCABaseline(patch_height=8, patch_width=8,
                                  channel_names=[...], target_compression_ratio=32,
                                  incremental_batch_size=4096)
        model.fit(patch_batches_iterable, n_train_patches=...)
        coeffs = model.encode(patches)
        recon_patches = model.decode(coeffs)
        recon_map = model.reconstruct(normalized_map)  # полный цикл
        model.save(output_dir)
        loaded = PatchPCABaseline.load(output_dir)
    """

    def __init__(
        self,
        patch_height: int,
        patch_width: int,
        channel_names: list[str],
        n_components: int | None = None,
        target_compression_ratio: float | None = None,
        incremental_batch_size: int = 4096,
        dtype: str = "float32",
    ):
        if incremental_batch_size <= 0:
            raise ValueError("incremental_batch_size должен быть положительным")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channel_names = list(channel_names)
        self.channel_count = len(channel_names)
        self.patch_feature_count = self.channel_count * patch_height * patch_width
        self.target_compression_ratio = target_compression_ratio
        self._requested_n_components = n_components
        self.incremental_batch_size = incremental_batch_size
        self.dtype = np.dtype(dtype)

        self.n_components: int | None = None
        self._pca: IncrementalPCA | None = None
        self.n_samples_seen_: int = 0

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #
    def fit(self, patch_batches: Iterable[np.ndarray], n_train_patches: int) -> "PatchPCABaseline":
        """Обучает IncrementalPCA на потоке батчей патчей.

        Args:
            patch_batches: итератор/генератор, отдающий последовательные
                порции патчей формы [n_in_batch, patch_feature_count].
                Все патчи одновременно в память НЕ загружаются — это
                гарантируется тем, что вызывающий код (CLI) сам является
                генератором по timestamp'ам.
            n_train_patches: общее число train-патчей (нужно заранее для
                валидации n_components, п.4 задачи).
        """
        self.n_components = resolve_n_components(
            patch_feature_count=self.patch_feature_count,
            n_train_patches=n_train_patches,
            target_compression_ratio=self.target_compression_ratio,
            n_components=self._requested_n_components,
        )
        if self.incremental_batch_size < self.n_components:
            raise ValueError(
                f"incremental_batch_size ({self.incremental_batch_size}) должен быть "
                f">= n_components ({self.n_components})"
            )

        self._pca = IncrementalPCA(n_components=self.n_components)
        buffer = np.empty((0, self.patch_feature_count), dtype=self.dtype)
        n_seen = 0

        for batch in patch_batches:
            batch = np.asarray(batch, dtype=self.dtype)
            if batch.ndim != 2 or batch.shape[1] != self.patch_feature_count:
                raise ValueError(
                    f"Батч патчей должен иметь форму [N, {self.patch_feature_count}], получено {batch.shape}"
                )
            buffer = np.concatenate([buffer, batch], axis=0)
            # Копим буфер, пока после отделения одного incremental_batch_size
            # порции остаток гарантированно >= n_components (или будет пуст) —
            # тем самым partial_fit никогда не получит "хвост" меньше n_components.
            while len(buffer) >= self.incremental_batch_size + self.n_components:
                chunk, buffer = buffer[: self.incremental_batch_size], buffer[self.incremental_batch_size :]
                self._pca.partial_fit(chunk)
                n_seen += len(chunk)

        if len(buffer) >= self.n_components:
            self._pca.partial_fit(buffer)
            n_seen += len(buffer)
        elif len(buffer) > 0:
            raise ValueError(
                f"Недостаточно данных для завершающего partial_fit: в буфере осталось "
                f"{len(buffer)} патчей, а n_components={self.n_components}. "
                f"Увеличьте объём train-данных, max_train_timestamps/max_patches_per_timestamp "
                f"или уменьшите n_components/incremental_batch_size."
            )

        self.n_samples_seen_ = n_seen
        return self

    # ------------------------------------------------------------------ #
    # Encode / decode / reconstruct
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("Модель не обучена: вызовите fit() или load() перед encode/decode")

    def encode(self, patches: np.ndarray) -> np.ndarray:
        """[n_patches, patch_feature_count] -> [n_patches, n_components]."""
        self._check_fitted()
        return self._pca.transform(np.asarray(patches, dtype=self.dtype))

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """[n_patches, n_components] -> [n_patches, patch_feature_count]."""
        self._check_fitted()
        return self._pca.inverse_transform(coefficients).astype(self.dtype)

    def reconstruct(self, normalized_map: np.ndarray) -> tuple[np.ndarray, PatchGridInfo]:
        """Полный цикл: карта -> патчи -> encode -> decode -> карта.

        Args:
            normalized_map: массив [batch, channel, height, width], уже
                нормализованный (см. normalization.py). Padding добавляется
                и удаляется автоматически внутри extract/reconstruct_from_patches.

        Returns:
            (reconstructed_map, grid) — grid можно переиспользовать, чтобы
            не пересчитывать геометрию патчей повторно.
        """
        patches, grid = extract_patches(normalized_map, self.patch_height, self.patch_width)
        coefficients = self.encode(patches)
        decoded_patches = self.decode(coefficients)
        reconstructed = reconstruct_from_patches(decoded_patches, grid)
        return reconstructed, grid

    # ------------------------------------------------------------------ #
    # Коэффициенты сжатия
    # ------------------------------------------------------------------ #
    def compression_info(
        self,
        original_shape: tuple[int, ...],
        n_patches: int,
        normalization_bytes: int = 0,
        metadata_bytes: int = 0,
    ) -> CompressionInfo:
        """Раздел 12 PCA_BASELINE.md: payload и end-to-end compression ratio.

        original_bytes считается по исходной (не padded) карте — именно то,
        что реально нужно было бы передать/хранить без сжатия.
        """
        self._check_fitted()
        item_size = self.dtype.itemsize
        original_bytes = int(np.prod(original_shape)) * item_size
        coefficients_bytes = n_patches * self.n_components * item_size
        model_bytes = self.model_size_bytes()

        payload_ratio = self.patch_feature_count / self.n_components
        denominator = coefficients_bytes + model_bytes + normalization_bytes + metadata_bytes
        end_to_end_ratio = original_bytes / denominator if denominator > 0 else float("inf")

        return CompressionInfo(
            original_shape=tuple(original_shape),
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            patch_feature_count=self.patch_feature_count,
            n_components=self.n_components,
            n_patches=n_patches,
            original_bytes=original_bytes,
            coefficients_bytes=coefficients_bytes,
            model_bytes=model_bytes,
            normalization_bytes=normalization_bytes,
            metadata_bytes=metadata_bytes,
            target_compression_ratio=self.target_compression_ratio,
            payload_compression_ratio=payload_ratio,
            end_to_end_compression_ratio=end_to_end_ratio,
        )

    def model_size_bytes(self) -> int:
        """Оценка размера сохранённой PCA-модели в байтах: components + mean
        (два главных по объёму поля; singular_values/explained_variance —
        малы по сравнению с components, но тоже учитываются)."""
        self._check_fitted()
        item_size = self.dtype.itemsize
        components_bytes = self._pca.components_.size * item_size
        mean_bytes = self._pca.mean_.size * item_size
        aux_bytes = (
            self._pca.explained_variance_.size
            + self._pca.explained_variance_ratio_.size
            + self._pca.singular_values_.size
        ) * item_size
        return int(components_bytes + mean_bytes + aux_bytes)

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #
    def save(self, output_dir: str | Path) -> Path:
        """Сохраняет pca_model.npz (раздел 13 задачи / раздел 8.1 PCA_BASELINE.md).
        Сохраняются только числовые параметры модели — никаких Dataset-объектов
        или произвольных python-объектов (явное требование п.13 задачи).
        """
        self._check_fitted()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / _MODEL_FILE
        np.savez(
            path,
            components=self._pca.components_,
            mean=self._pca.mean_,
            explained_variance=self._pca.explained_variance_,
            explained_variance_ratio=self._pca.explained_variance_ratio_,
            singular_values=self._pca.singular_values_,
            n_components=np.asarray(self.n_components),
            n_samples_seen=np.asarray(self.n_samples_seen_),
            patch_height=np.asarray(self.patch_height),
            patch_width=np.asarray(self.patch_width),
            channel_count=np.asarray(self.channel_count),
            channel_names=np.asarray(self.channel_names, dtype=object),
            feature_count=np.asarray(self.patch_feature_count),
            target_compression_ratio=np.asarray(
                self.target_compression_ratio if self.target_compression_ratio is not None else np.nan
            ),
            dtype=np.asarray(str(self.dtype)),
        )
        return path

    @classmethod
    def load(cls, model_dir: str | Path) -> "PatchPCABaseline":
        """Загружает модель без повторного fit (раздел 13 задачи: "Artifact
        должен содержать достаточно информации для восстановления модели
        без повторной подгонки")."""
        model_dir = Path(model_dir)
        path = model_dir / _MODEL_FILE if model_dir.is_dir() else model_dir
        with np.load(path, allow_pickle=True) as npz:
            channel_names = list(npz["channel_names"])
            target_ratio = float(npz["target_compression_ratio"])
            target_ratio = None if np.isnan(target_ratio) else target_ratio

            model = cls(
                patch_height=int(npz["patch_height"]),
                patch_width=int(npz["patch_width"]),
                channel_names=channel_names,
                n_components=int(npz["n_components"]),
                target_compression_ratio=target_ratio,
                incremental_batch_size=max(int(npz["n_components"]), 1),
                dtype=str(npz["dtype"]),
            )
            pca = IncrementalPCA(n_components=int(npz["n_components"]))
            pca.components_ = npz["components"]
            pca.mean_ = npz["mean"]
            pca.explained_variance_ = npz["explained_variance"]
            pca.explained_variance_ratio_ = npz["explained_variance_ratio"]
            pca.singular_values_ = npz["singular_values"]
            pca.n_components_ = int(npz["n_components"])
            pca.n_samples_seen_ = int(npz["n_samples_seen"])
            pca.n_features_in_ = int(npz["feature_count"])

            model._pca = pca
            model.n_components = int(npz["n_components"])
            model.n_samples_seen_ = int(npz["n_samples_seen"])
        return model