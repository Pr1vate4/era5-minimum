import pytest
import numpy as np

from era5_minimum.baselines.dataset import generate_synthetic_dataset, temporal_split
from era5_minimum.baselines.normalization import ChannelNormalizer, normalize, denormalize
from era5_minimum.baselines.patches import extract_patches, reconstruct_from_patches
from era5_minimum.baselines.patch_pca import PatchPCABaseline
from era5_minimum.baselines.metrics import compute_metrics, MetricsResult
from era5_minimum.baselines.artifacts import (
    build_summary_artifact,
    build_experiment_artifact,
    build_reconstruction_artifact,
)


# ==============================================================================
# Общая фикстура: подготовка данных и обучение модели
# ==============================================================================
@pytest.fixture(scope="module")
def pipeline_setup():
    """
    Генерирует синтетический датасет, разбивает его, обучает нормализацию и PCA.
    Возвращает все необходимые объекты для сквозных тестов.
    """
    # 1. Генерация данных (размер НЕ кратен 8x8, чтобы проверить padding)
    # height=17, width=33 -> pad до 24x40 -> 3x5 = 15 патчей на timestamp
    ds = generate_synthetic_dataset(n_timestamps=24, height=17, width=33, seed=42)

    # 2. Temporal split (50% train, 25% val, 25% test)
    split = temporal_split(ds, train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25)

    # 3. Нормализация (считаем ТОЛЬКО по train, учитываем sst_mask)
    norm = ChannelNormalizer(ds.channel_names)
    norm.update(split.train.data, valid_mask=split.train.sst_mask)
    stats = norm.finalize()

    # 4. Подготовка данных для PCA
    # sklearn.IncrementalPCA не умеет работать с NaN. Заменяем их на 0.0.
    # При этом sst_mask всё равно будет передан в метрики, чтобы "суша"
    # не участвовала в расчёте MSE/MAE (требование п.6.2 ТЗ).
    train_data_clean = np.nan_to_num(split.train.data, nan=0.0)
    test_data_clean = np.nan_to_num(split.test.data, nan=0.0)

    # 5. Обучение PCA
    # patch_feature_count = 8 каналов * 8 * 8 = 512
    # target_compression_ratio = 16.0 -> n_components = 512 // 16 = 32
    model = PatchPCABaseline(
        patch_height=8, patch_width=8,
        channel_names=ds.channel_names,
        target_compression_ratio=16.0,
        incremental_batch_size=64  # должно быть >= n_components (32)
    )

    train_norm = normalize(train_data_clean, stats)
    train_patches, _ = extract_patches(train_norm, 8, 8)

    # 12 train timestamps * 15 patches = 180 patches
    model.fit([train_patches], n_train_patches=train_patches.shape[0])

    return {
        "ds": ds,
        "split": split,
        "stats": stats,
        "model": model,
        "test_data_clean": test_data_clean,
    }


# ==============================================================================
# ЧАСТЬ 1. Минимальный pipeline
# ==============================================================================
class TestPart1MinimalPipeline:
    """Полностью воспроизводит сценарий из ТЗ без проверки качества."""

    def test_minimal_pipeline_no_exceptions(self, pipeline_setup):
        """Проверяется, что полный цикл выполняется без исключений."""
        ds = pipeline_setup["ds"]
        stats = pipeline_setup["stats"]
        model = pipeline_setup["model"]
        test_data_clean = pipeline_setup["test_data_clean"]

        # Pipeline
        norm_data = normalize(test_data_clean, stats)
        patches, grid = extract_patches(norm_data, 8, 8)
        coeffs = model.encode(patches)
        decoded_patches = model.decode(coeffs)
        recon_norm = reconstruct_from_patches(decoded_patches, grid)
        recon_data = denormalize(recon_norm, stats)

        # Проверка корректных размеров результата
        assert recon_data.shape == test_data_clean.shape
        assert recon_data.shape == (6, 8, 17, 33)  # 25% от 24 timestamps = 6


# ==============================================================================
# ЧАСТЬ 2. Проверка reconstruction
# ==============================================================================
class TestPart2ReconstructionChecks:
    """Проверка требований smoke-проверки ТЗ к выходным данным."""

    def test_reconstruction_shape_and_finite(self, pipeline_setup):
        """Совпадение формы, отсутствие NaN и Inf."""
        ds = pipeline_setup["ds"]
        stats = pipeline_setup["stats"]
        model = pipeline_setup["model"]
        test_data_clean = pipeline_setup["test_data_clean"]

        # Выполняем pipeline
        norm_data = normalize(test_data_clean, stats)
        patches, grid = extract_patches(norm_data, 8, 8)
        coeffs = model.encode(patches)
        decoded_patches = model.decode(coeffs)
        recon_norm = reconstruct_from_patches(decoded_patches, grid)
        recon_data = denormalize(recon_norm, stats)

        # 1. Совпадение формы
        assert recon_data.shape == test_data_clean.shape

        # 2. Отсутствие NaN
        assert not np.any(np.isnan(recon_data)), "В реконструкции обнаружены NaN"

        # 3. Отсутствие Inf
        assert not np.any(np.isinf(recon_data)), "В реконструкции обнаружены Inf"


# ==============================================================================
# ЧАСТЬ 3. Метрики
# ==============================================================================
class TestPart3Metrics:
    """Проверка вычисления итоговых метрик с учётом масок."""

    def test_metrics_computation(self, pipeline_setup):
        """MSE/MAE/RMSE считаются, конечны, исключения отсутствуют."""
        ds = pipeline_setup["ds"]
        split = pipeline_setup["split"]
        stats = pipeline_setup["stats"]
        model = pipeline_setup["model"]
        test_data_clean = pipeline_setup["test_data_clean"]

        # Получаем реконструкцию
        norm_data = normalize(test_data_clean, stats)
        patches, grid = extract_patches(norm_data, 8, 8)
        coeffs = model.encode(patches)
        decoded_patches = model.decode(coeffs)
        recon_norm = reconstruct_from_patches(decoded_patches, grid)
        recon_data = denormalize(recon_norm, stats)

        # Считаем метрики.
        # ВАЖНО: передаём ОРИГИНАЛЬНЫЕ данные (с NaN) и sst_mask.
        # Функция compute_metrics сама исключит невалидные точки (п.6.2 ТЗ).
        result = compute_metrics(
            original=split.test.data,
            reconstruction=recon_data,
            channel_names=ds.channel_names,
            valid_mask=split.test.sst_mask
        )

        # Проверки
        assert isinstance(result, MetricsResult)

        # Конечность всех значений (п.19.30 ТЗ)
        result.assert_finite()  # Бросит ValueError, если хоть одна метрика NaN/Inf

        # Наличие базовых метрик
        assert "mse" in result.overall
        assert "mae" in result.overall
        assert "rmse" in result.overall

        # Проверка, что valid_point_count > 0 (маска сработала)
        assert result.valid_point_count > 0


# ==============================================================================
# ЧАСТЬ 4. Итоговая проверка baseline и артефактов
# ==============================================================================
class TestPart4FinalBaselineAndArtifacts:
    """Финальная проверка: обучение, реконструкция, артефакты, метрики."""

    def test_full_baseline_and_artifacts(self, pipeline_setup, tmp_path):
        """Запускается полный pipeline и проверяется генерация JSON-артефактов."""
        ds = pipeline_setup["ds"]
        split = pipeline_setup["split"]
        stats = pipeline_setup["stats"]
        model = pipeline_setup["model"]
        test_data_clean = pipeline_setup["test_data_clean"]

        # 1. Сохранение и загрузка модели (проверка сериализации)
        model.save(tmp_path)
        loaded_model = PatchPCABaseline.load(tmp_path)
        assert loaded_model.n_components == model.n_components

        # 2. Получение реконструкции
        norm_data = normalize(test_data_clean, stats)
        patches, grid = extract_patches(norm_data, 8, 8)
        coeffs = loaded_model.encode(patches)
        decoded_patches = loaded_model.decode(coeffs)
        recon_norm = reconstruct_from_patches(decoded_patches, grid)
        recon_data = denormalize(recon_norm, stats)

        # 3. Расчёт метрик
        metrics_result = compute_metrics(
            original=split.test.data,
            reconstruction=recon_data,
            channel_names=ds.channel_names,
            valid_mask=split.test.sst_mask
        )

        # 4. Генерация артефактов (контракт ARTIFACT_API_CONTRACT.md)
        summary = build_summary_artifact(
            experiment_count=1, completed=1, failed=0, is_demo=True
        )

        comp_info = loaded_model.compression_info(
            original_shape=split.test.data.shape,
            n_patches=patches.shape[0]
        )

        experiment = build_experiment_artifact(
            experiment_id="exp_smoke_001",
            name="Smoke Test PCA",
            status="completed",
            is_demo=True,
            compression_type="pca_payload",
            payload_ratio=comp_info.payload_compression_ratio,
            sample_count=metrics_result.valid_point_count,
            selection_strategy="temporal_train",
            metrics=metrics_result.overall,
        )

        # Берём первый timestamp и первый канал для артефакта реконструкции
        t_idx, c_idx = 0, 0
        channel_name = ds.channel_names[c_idx]
        timestamp_str = str(ds.timestamps[t_idx])

        original_slice = split.test.data[t_idx, c_idx]
        recon_slice = recon_data[t_idx, c_idx]
        abs_error = np.abs(original_slice - recon_slice)

        reconstruction_art = build_reconstruction_artifact(
            experiment_id="exp_smoke_001",
            channel=channel_name,
            timestamp=timestamp_str,
            latitude=ds.latitude,
            longitude=ds.longitude,
            original=original_slice,
            reconstruction=recon_slice,
            absolute_error=abs_error,
            is_demo=True,
        )

        # 5. Финальные проверки артефактов
        assert summary["is_demo"] is True
        assert experiment["model"] == "patch_pca"  # Жёсткое требование п.17 ТЗ
        assert experiment["compression"]["ratio"] > 0

        # Все метрики в артефакте должны быть конечными
        for metric_name, value in experiment["metrics"].items():
            assert np.isfinite(value), f"Метрика {metric_name} не конечна!"

        # Проверка геометрии в артефакте реконструкции
        assert len(reconstruction_art["latitude"]) == original_slice.shape[0]
        assert len(reconstruction_art["longitude"]) == original_slice.shape[1]
        assert reconstruction_art["units"] == "m s**-1" if channel_name in ["u10", "v10"] else True