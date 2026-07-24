"""
CLI для обучения Patch-based PCA baseline (раздел 14 PCA_BASELINE.md).
Выполняет полный цикл из 17 шагов, описанных в задаче:
    1. читает config;
    2. находит входные NPZ (или генерирует synthetic при --smoke-test);
    3. проверяет метаданные (порядок каналов, units и т.д. — внутри
       dataset.load_era5_npz_dataset);
    4. сортирует timestamps (тоже внутри load_era5_npz_dataset);
    5. проверяет дубликаты (там же);
    6. делает temporal split (dataset.temporal_split);
    7. считает train-only нормализацию (normalization.ChannelNormalizer);
    8. определяет n_components (patch_pca.resolve_n_components, внутри model.fit);
    9. подгоняет IncrementalPCA порциями (model.fit с генератором батчей);
    10-11. сохраняет PCA artifact и normalization.json;
    12. выполняет validation reconstruction;
    13-14. считает метрики и explained variance;
    15. считает payload и end-to-end ratio;
    16-17. сохраняет resolved config и fit summary.

Пример:
    python scripts/fit_pca_baseline.py \\
        --config configs/patch_pca_32x.yaml \\
        --input outputs/real_era5/week_2024_01_02_2024_01_08 \\
        --output-dir outputs/models/patch_pca_32x

Smoke-режим (без интернета и реальных данных):
    python scripts/fit_pca_baseline.py \\
        --config configs/patch_pca_32x.yaml \\
        --smoke-test \\
        --output-dir /tmp/era5-pca-smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.baselines.dataset import (  # noqa: E402
    load_era5_npz_dataset,
    temporal_split,
    generate_synthetic_dataset,
)
from era5_minimum.baselines.normalization import ChannelNormalizer, normalize  # noqa: E402
from era5_minimum.baselines.patches import extract_patches, patch_grid_dims  # noqa: E402
from era5_minimum.baselines.patch_pca import PatchPCABaseline  # noqa: E402
from era5_minimum.baselines.metrics import compute_metrics  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение Patch-based PCA baseline (ERA5-Minimum)")
    parser.add_argument("--config", type=str, required=True, help="Путь к YAML-конфигу")
    parser.add_argument("--input", type=str, default=None, help="Путь к NPZ-файлу или директории с NPZ")
    parser.add_argument("--output-dir", type=str, default=None, help="Куда сохранить артефакты модели")
    parser.add_argument("--target-ratio", type=float, default=None, help="Override target_compression_ratio")
    parser.add_argument("--n-components", type=int, default=None, help="Override n_components")
    parser.add_argument("--patch-height", type=int, default=None)
    parser.add_argument("--patch-width", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--max-train-timestamps", type=int, default=None)
    parser.add_argument("--max-patches-per-timestamp", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Synthetic dataset вместо реальных данных, полный цикл fit/encode/decode/metrics на CPU без интернета",
    )
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config or {}


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """CLI-флаги перекрывают значения из YAML — приоритет у явных флагов."""
    overrides = {
        "input": args.input,
        "output_dir": args.output_dir,
        "target_compression_ratio": args.target_ratio,
        "n_components": args.n_components,
        "patch_height": args.patch_height,
        "patch_width": args.patch_width,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "max_train_timestamps": args.max_train_timestamps,
        "max_patches_per_timestamp": args.max_patches_per_timestamp,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _iter_train_patch_batches(
    normalized_data: np.ndarray,
    patch_height: int,
    patch_width: int,
    max_patches_per_timestamp: int | None,
    seed: int,
) -> Iterator[np.ndarray]:
    """Генератор батчей патчей: по одному timestamp за раз, чтобы весь
    train-набор патчей никогда не находился в памяти одновременно
    (требование "данные не требуется полностью держать в RAM").
    """
    rng = np.random.default_rng(seed)
    for t in range(normalized_data.shape[0]):
        single = normalized_data[t : t + 1]  # [1, channel, height, width]
        patches, _grid = extract_patches(single, patch_height, patch_width)
        if max_patches_per_timestamp is not None and len(patches) > max_patches_per_timestamp:
            idx = rng.choice(len(patches), size=max_patches_per_timestamp, replace=False)
            patches = patches[idx]
        yield patches


def count_train_patches(
    n_timestamps: int, height: int, width: int, patch_height: int, patch_width: int, max_patches_per_timestamp: int | None
) -> int:
    n_patches_h, n_patches_w = patch_grid_dims(height, width, patch_height, patch_width)
    per_timestamp = n_patches_h * n_patches_w
    if max_patches_per_timestamp is not None:
        per_timestamp = min(per_timestamp, max_patches_per_timestamp)
    return n_timestamps * per_timestamp


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args)

    seed = int(config.get("seed", 0))
    np.random.seed(seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2-5. Загрузка данных (или synthetic dataset для smoke-test).
    if args.smoke_test:
        dataset = generate_synthetic_dataset(seed=seed)
        is_demo = True
    else:
        if not config.get("input"):
            raise SystemExit("Нужно указать --input (или input: в конфиге), если не используется --smoke-test")
        dataset = load_era5_npz_dataset(config["input"])
        is_demo = False

    expected_channels = config.get("expected_channel_names")
    if expected_channels and list(expected_channels) != dataset.channel_names:
        raise SystemExit(
            f"channel_names датасета {dataset.channel_names} не совпадает с "
            f"expected_channel_names из конфига {list(expected_channels)}"
        )

    max_train_ts = config.get("max_train_timestamps")

    # 6. Temporal split.
    split = temporal_split(
        dataset,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        test_fraction=float(config["test_fraction"]),
    )
    train = split.train
    if max_train_ts is not None:
        train.data = train.data[:max_train_ts]
        train.sst_mask = train.sst_mask[:max_train_ts]
        train.timestamps = train.timestamps[:max_train_ts]

    # 7. Train-only нормализация (streaming, с учётом SST mask и NaN).
    normalizer = ChannelNormalizer(train.channel_names)
    sst_index = train.channel_names.index("sst") if "sst" in train.channel_names else None
    if sst_index is not None:
        full_valid_mask = np.ones_like(train.data, dtype=bool)
        full_valid_mask[:, sst_index, :, :] = train.sst_mask.astype(bool)
        normalizer.update(train.data, valid_mask=full_valid_mask)
    else:
        normalizer.update(train.data)
    norm_stats = normalizer.finalize(source_split="train")

    train_normalized = normalize(train.data, norm_stats).astype(config.get("dtype", "float32"))

    patch_height = int(config["patch_height"])
    patch_width = int(config["patch_width"])
    max_patches_per_ts = config.get("max_patches_per_timestamp")

    n_train_patches = count_train_patches(
        train_normalized.shape[0],
        train.data.shape[2],
        train.data.shape[3],
        patch_height,
        patch_width,
        max_patches_per_ts,
    )

    # 8-9. Определение n_components и подгонка IncrementalPCA порциями.
    model = PatchPCABaseline(
        patch_height=patch_height,
        patch_width=patch_width,
        channel_names=train.channel_names,
        n_components=config.get("n_components"),
        target_compression_ratio=config.get("target_compression_ratio"),
        incremental_batch_size=int(config.get("incremental_batch_size", 4096)),
        dtype=config.get("dtype", "float32"),
    )
    batch_iterator = _iter_train_patch_batches(
        train_normalized, patch_height, patch_width, max_patches_per_ts, seed
    )
    model.fit(batch_iterator, n_train_patches=n_train_patches)

    # 10-11. Сохранение PCA artifact и normalization.
    model.save(output_dir)
    norm_stats.save(output_dir / "normalization.json")

    # 12-14. Validation reconstruction + метрики + explained variance.
    validation = split.validation if split.validation.n_timestamps > 0 else train
    val_normalized = normalize(validation.data, norm_stats).astype(config.get("dtype", "float32"))
    reconstructed_normalized, _grid = model.reconstruct(val_normalized)
    from era5_minimum.baselines.normalization import denormalize  # локальный импорт для читаемости

    reconstructed = denormalize(reconstructed_normalized, norm_stats)

    valid_mask = None
    if sst_index is not None:
        valid_mask = np.ones_like(validation.data, dtype=bool)
        valid_mask[:, sst_index, :, :] = validation.sst_mask.astype(bool)
    metrics_result = compute_metrics(
        validation.data, reconstructed, validation.channel_names, valid_mask=valid_mask
    )
    metrics_result.assert_finite()

    # 15. Payload / end-to-end compression ratio.
    normalization_bytes = norm_stats.mean.nbytes + norm_stats.std.nbytes + norm_stats.valid_count.nbytes
    metadata_bytes = 512  # грубая оценка размера служебных JSON-метаданных
    compression = model.compression_info(
        original_shape=train.data.shape[1:],
        n_patches=n_train_patches,
        normalization_bytes=int(normalization_bytes),
        metadata_bytes=metadata_bytes,
    )

    # 16-17. resolved config + fit summary.
    resolved_config = dict(config)
    resolved_config["n_components_resolved"] = model.n_components
    resolved_config["patch_feature_count"] = model.patch_feature_count
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved_config, fh, allow_unicode=True, sort_keys=False)

    fit_summary = {
        "is_demo": is_demo,
        "model": "patch_pca",
        "n_train_timestamps": int(train.n_timestamps),
        "n_validation_timestamps": int(validation.n_timestamps),
        "n_train_patches": int(n_train_patches),
        "n_components": model.n_components,
        "explained_variance_ratio_sum": float(np.sum(model._pca.explained_variance_ratio_)),
        "validation_metrics": metrics_result.to_json_dict(),
        "compression": compression.to_json_dict(),
    }
    with open(output_dir / "fit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(fit_summary, fh, ensure_ascii=False, indent=2)

    print(f"Готово. n_components={model.n_components}, "
          f"payload_ratio={compression.payload_compression_ratio:.2f}x, "
          f"end_to_end_ratio={compression.end_to_end_compression_ratio:.2f}x")
    print(f"Артефакты сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())