"""
Точка входа пакета `era5_minimum.baselines`. Собирает и реэкспортирует
публичный API baseline-модели Patch-based PCA, чтобы остальной код
проекта (CLI-скрипты, тесты, будущий ConvAE-код) мог писать:

    from era5_minimum.baselines import PatchPCABaseline, extract_patches

вместо длинных путей импорта до конкретных модулей.

Ничего "исполняемого" в этом файле нет — только реэкспорт.
"""

from era5_minimum.baselines.patches import (
    PatchGridInfo,
    compute_padding,
    patch_grid_dims,
    extract_patches,
    reconstruct_from_patches,
)
from era5_minimum.baselines.normalization import (
    ChannelNormalizer,
    NormalizationStats,
)
from era5_minimum.baselines.metrics import compute_metrics, MetricsResult
from era5_minimum.baselines.dataset import (
    ERA5NPZDataset,
    TemporalSplit,
    load_era5_npz_dataset,
    temporal_split,
    generate_synthetic_dataset,
)
from era5_minimum.baselines.patch_pca import PatchPCABaseline, resolve_n_components, CompressionInfo
from era5_minimum.baselines.artifacts import (
    CANONICAL_CHANNELS,
    FORBIDDEN_CHANNEL_NAMES,
    build_summary_artifact,
    build_experiment_artifact,
    build_reconstruction_artifact,
    validate_channel_name,
)

__all__ = [
    "PatchGridInfo",
    "compute_padding",
    "patch_grid_dims",
    "extract_patches",
    "reconstruct_from_patches",
    "ChannelNormalizer",
    "NormalizationStats",
    "compute_metrics",
    "MetricsResult",
    "ERA5NPZDataset",
    "TemporalSplit",
    "load_era5_npz_dataset",
    "temporal_split",
    "generate_synthetic_dataset",
    "PatchPCABaseline",
    "resolve_n_components",
    "CompressionInfo",
    "CANONICAL_CHANNELS",
    "FORBIDDEN_CHANNEL_NAMES",
    "build_summary_artifact",
    "build_experiment_artifact",
    "build_reconstruction_artifact",
    "validate_channel_name",
]