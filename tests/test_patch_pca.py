import pytest
import numpy as np
from pathlib import Path

from era5_minimum.baselines.patch_pca import PatchPCABaseline, resolve_n_components, CompressionInfo


# ==============================================================================
# ЧАСТЬ 1: Конструктор и параметры модели
# ==============================================================================
class TestPart1Constructor:
    """Проверяется корректность создания PatchPCABaseline и валидация параметров."""

    def test_init_with_n_components(self):
        model = PatchPCABaseline(
            patch_height=4, patch_width=4,
            channel_names=["t", "u"],
            n_components=10,
        )
        assert model.patch_height == 4
        assert model.patch_width == 4
        assert model.channel_count == 2
        assert model.patch_feature_count == 2 * 4 * 4  # 32
        assert model._requested_n_components == 10
        assert model.target_compression_ratio is None

    def test_init_with_target_ratio(self):
        model = PatchPCABaseline(
            patch_height=2, patch_width=2,
            channel_names=["c1"],
            target_compression_ratio=8.0,
        )
        assert model.target_compression_ratio == 8.0
        assert model._requested_n_components is None

    def test_init_defaults(self):
        model = PatchPCABaseline(
            patch_height=2, patch_width=2,
            channel_names=["c1"],
        )
        assert model.incremental_batch_size == 4096
        assert model.dtype == np.dtype("float32")
        assert model.n_components is None
        assert model._pca is None

    def test_init_invalid_batch_size(self):
        with pytest.raises(ValueError, match="incremental_batch_size должен быть положительным"):
            PatchPCABaseline(
                patch_height=2, patch_width=2,
                channel_names=["c1"],
                incremental_batch_size=0,
            )
        with pytest.raises(ValueError, match="incremental_batch_size должен быть положительным"):
            PatchPCABaseline(
                patch_height=2, patch_width=2,
                channel_names=["c1"],
                incremental_batch_size=-10,
            )


# ==============================================================================
# ЧАСТЬ 2: resolve_n_components()
# ==============================================================================
class TestPart2ResolveNComponents:
    """Unit-тесты для функции resolve_n_components. Каждая ветка — отдельный тест."""

    def test_explicit_n_components(self):
        res = resolve_n_components(
            patch_feature_count=100, n_train_patches=1000, n_components=20
        )
        assert res == 20

    def test_target_ratio(self):
        # 100 // 4 = 25
        res = resolve_n_components(
            patch_feature_count=100, n_train_patches=1000,
            target_compression_ratio=4.0,
        )
        assert res == 25

    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="Нужно задать либо n_components"):
            resolve_n_components(patch_feature_count=100, n_train_patches=1000)

    def test_negative_ratio_raises(self):
        with pytest.raises(ValueError, match="target_compression_ratio должен быть положительным"):
            resolve_n_components(
                patch_feature_count=100, n_train_patches=1000,
                target_compression_ratio=-1.0,
            )

    def test_zero_n_components_raises(self):
        with pytest.raises(ValueError, match="n_components должен быть положительным"):
            resolve_n_components(
                patch_feature_count=100, n_train_patches=1000, n_components=0
            )

    def test_n_components_gt_feature_count_raises(self):
        with pytest.raises(ValueError, match="не может быть больше patch_feature_count"):
            resolve_n_components(
                patch_feature_count=10, n_train_patches=1000, n_components=15
            )

    def test_n_components_gt_train_patches_raises(self):
        with pytest.raises(ValueError, match="не может быть больше числа train-патчей"):
            resolve_n_components(
                patch_feature_count=100, n_train_patches=5, n_components=10
            )

    def test_correct_calculation_with_ratio_clamp_to_one(self):
        # feature_count=5, ratio=10 -> 5//10 = 0 -> max(1, 0) = 1
        res = resolve_n_components(
            patch_feature_count=5, n_train_patches=100,
            target_compression_ratio=10.0,
        )
        assert res == 1


# ==============================================================================
# ЧАСТЬ 3: Обучение (fit)
# ==============================================================================
class TestPart3Fit:
    """Проверяется выполнение требований ТЗ по IncrementalPCA и буферизации.

    Важно: patch_feature_count = channel_count * patch_h * patch_w
    должно быть СТРОГО >= n_components, а incremental_batch_size >= n_components.

    Здесь: channel_names=["c1","c2"], patch=2x2 → feature_count = 2*2*2 = 8.
    n_components=5 <= 8 ✓, batch_size=10 >= 5 ✓.
    """

    FEATURE_COUNT = 8  # 2 канала × 2 × 2

    def _make_model(self, n_comp=5, batch_size=10):
        return PatchPCABaseline(
            patch_height=2, patch_width=2,
            channel_names=["c1", "c2"],
            n_components=n_comp,
            incremental_batch_size=batch_size,
        )

    def test_fit_single_large_batch(self):
        model = self._make_model(n_comp=5, batch_size=10)
        data = np.random.rand(20, self.FEATURE_COUNT).astype(np.float32)
        model.fit([data], n_train_patches=20)
        assert model.n_samples_seen_ == 20


    def test_fit_multiple_small_batches(self):
        model = self._make_model(n_comp=5, batch_size=10)
        batches = [
            np.random.rand(5, self.FEATURE_COUNT).astype(np.float32)
            for _ in range(4)
        ]  # всего 20 патчей
        model.fit(batches, n_train_patches=20)
        assert model.n_samples_seen_ == 20


    def test_fit_remainder_less_than_n_components_raises(self):
        """Вызывающий код заявил n_train_patches=20, но фактически подал
        только 3 патча → после while-цикла буфер (3) < n_components (5)."""
        model = self._make_model(n_comp=5, batch_size=10)
        data = np.random.rand(3, self.FEATURE_COUNT).astype(np.float32)
        with pytest.raises(ValueError, match="Недостаточно данных для завершающего partial_fit"):
            model.fit([data], n_train_patches=20)  # завышено!


    def test_fit_remainder_equals_n_components(self):
        """15 патчей, batch_size=10, n_components=5.
        while: 15 >= 10+5=15 → chunk=10, buf=5.
        Итог: partial_fit(10), partial_fit(5)."""
        model = self._make_model(n_comp=5, batch_size=10)
        data = np.random.rand(15, self.FEATURE_COUNT).astype(np.float32)
        model.fit([data], n_train_patches=15)
        assert model.n_samples_seen_ == 15


    def test_fit_multiple_partial_fit_calls(self):
        """35 патчей, batch_size=10, n_components=5.
        while: 35→25→15→5 (три chunk по 10), затем partial_fit(5).
        Итого 4 вызова partial_fit, n_seen=35."""
        model = self._make_model(n_comp=5, batch_size=10)
        data = np.random.rand(35, self.FEATURE_COUNT).astype(np.float32)
        model.fit([data], n_train_patches=35)
        assert model.n_samples_seen_ == 35
        assert model._pca.n_samples_seen_ == 35


    def test_fit_sets_model_state(self):
        model = self._make_model(n_comp=5, batch_size=10)
        data = np.random.rand(20, self.FEATURE_COUNT).astype(np.float32)
        model.fit([data], n_train_patches=20)

        assert model._pca is not None
        assert model.n_components == 5
        assert model.n_samples_seen_ == 20


# ==============================================================================
# ЧАСТЬ 4: Encode / Decode
# ==============================================================================
class TestPart4EncodeDecode:
    """Проверяются основные операции модели (encode/decode)."""

    @pytest.fixture
    def trained_model(self):
        model = PatchPCABaseline(
            patch_height=2, patch_width=2,
            channel_names=["c1", "c2"],
            n_components=3,
            incremental_batch_size=10,
        )
        # feature_count = 2 * 2 * 2 = 8
        data = np.random.rand(20, 8).astype(np.float32)
        model.fit([data], n_train_patches=20)
        return model

    def test_encode_shape(self, trained_model):
        patches = np.random.rand(15, 8).astype(np.float32)
        coeffs = trained_model.encode(patches)
        assert coeffs.shape == (15, 3)
        assert coeffs.dtype == np.float64

    def test_decode_shape_and_dtype(self, trained_model):
        coeffs = np.random.rand(15, 3).astype(np.float64)
        patches = trained_model.decode(coeffs)
        assert patches.shape == (15, 8)
        # decode() явно делает .astype(self.dtype) → float32
        assert patches.dtype == np.float32

    def test_encode_decode_not_fitted_raises(self):
        model = PatchPCABaseline(
            patch_height=2, patch_width=2,
            channel_names=["c1"], n_components=2,
        )
        with pytest.raises(RuntimeError, match="Модель не обучена"):
            model.encode(np.zeros((5, 4)))
        with pytest.raises(RuntimeError, match="Модель не обучена"):
            model.decode(np.zeros((5, 2)))

    def test_compatibility_of_sizes(self, trained_model):
        original = np.random.rand(10, 8).astype(np.float32)
        coeffs = trained_model.encode(original)
        reconstructed = trained_model.decode(coeffs)
        assert reconstructed.shape == original.shape


# ==============================================================================
# ЧАСТЬ 5: Compression Info
# ==============================================================================
class TestPart5CompressionInfo:
    """Отдельный блок для проверки вычисления коэффициентов сжатия.

    channel_names=["c1","c2","c3"], patch=4x4 → feature_count = 3*4*4 = 48.
    n_components=16, incremental_batch_size=20 (>= 16 ✓).
    """

    @pytest.fixture
    def trained_model(self):
        model = PatchPCABaseline(
            patch_height=4, patch_width=4,
            channel_names=["c1", "c2", "c3"],
            n_components=16,
            incremental_batch_size=20,   # ← исправлено: было 10 < 16
        )
        # feature_count = 3 * 4 * 4 = 48
        data = np.random.rand(50, 48).astype(np.float32)
        model.fit([data], n_train_patches=50)
        return model

    def test_payload_ratio(self, trained_model):
        info = trained_model.compression_info(
            original_shape=(1, 3, 10, 10), n_patches=100
        )
        # payload = feature_count / n_components = 48 / 16 = 3.0
        assert info.payload_compression_ratio == 3.0

    def test_end_to_end_ratio(self, trained_model):
        original_shape = (1, 3, 10, 10)
        n_patches = 100
        info = trained_model.compression_info(
            original_shape=original_shape,
            n_patches=n_patches,
            normalization_bytes=100,
            metadata_bytes=50,
        )

        item_size = 4  # float32
        original_bytes = int(np.prod(original_shape)) * item_size
        coefficients_bytes = n_patches * 16 * item_size
        model_bytes = trained_model.model_size_bytes()

        expected_denominator = coefficients_bytes + model_bytes + 100 + 50
        expected_ratio = original_bytes / expected_denominator

        assert np.isclose(info.end_to_end_compression_ratio, expected_ratio)

    def test_bytes_calculations(self, trained_model):
        info = trained_model.compression_info(
            original_shape=(2, 3, 8, 8), n_patches=200
        )
        item_size = 4
        assert info.original_bytes == int(np.prod((2, 3, 8, 8))) * item_size
        assert info.coefficients_bytes == 200 * 16 * item_size

    def test_info_values_after_fit(self, trained_model):
        info = trained_model.compression_info(
            original_shape=(1, 3, 10, 10), n_patches=100
        )
        assert info.patch_feature_count == 48
        assert info.n_components == 16
        assert info.patch_height == 4
        assert info.patch_width == 4


# ==============================================================================
# ЧАСТЬ 6: Сохранение и загрузка
# ==============================================================================
class TestPart6SaveLoad:
    """Последний независимый блок: проверка сериализации и десериализации.

    channel_names=["ch_a","ch_b"], patch=2x2 → feature_count=8.
    n_components=4, incremental_batch_size=10 (>= 4 ✓).
    """

    @pytest.fixture
    def trained_model(self):
        model = PatchPCABaseline(
            patch_height=2, patch_width=2,
            channel_names=["ch_a", "ch_b"],
            n_components=4,
            target_compression_ratio=2.0,
            incremental_batch_size=10,
        )
        data = np.random.rand(30, 8).astype(np.float32)
        model.fit([data], n_train_patches=30)
        return model

    def test_save_creates_file(self, trained_model, tmp_path):
        saved_path = trained_model.save(tmp_path)
        assert saved_path.exists()
        assert saved_path.name == "pca_model.npz"

    def test_save_and_load_params(self, trained_model, tmp_path):
        trained_model.save(tmp_path)
        loaded_model = PatchPCABaseline.load(tmp_path)

        assert loaded_model.patch_height == trained_model.patch_height
        assert loaded_model.patch_width == trained_model.patch_width
        assert loaded_model.channel_count == trained_model.channel_count
        assert loaded_model.n_components == trained_model.n_components
        assert loaded_model.target_compression_ratio == pytest.approx(
            trained_model.target_compression_ratio
        )
        assert loaded_model.dtype == trained_model.dtype
        assert loaded_model.n_samples_seen_ == trained_model.n_samples_seen_

    def test_save_and_load_encode_decode(self, trained_model, tmp_path):
        trained_model.save(tmp_path)
        loaded_model = PatchPCABaseline.load(tmp_path)

        test_patches = np.random.rand(15, 8).astype(np.float32)

        # encode
        coeffs_orig = trained_model.encode(test_patches)
        coeffs_loaded = loaded_model.encode(test_patches)
        np.testing.assert_array_almost_equal(coeffs_orig, coeffs_loaded)

        # decode
        test_coeffs = np.random.rand(15, 4).astype(np.float64)
        patches_orig = trained_model.decode(test_coeffs)
        patches_loaded = loaded_model.decode(test_coeffs)
        np.testing.assert_array_almost_equal(patches_orig, patches_loaded)

    def test_load_from_file_path(self, trained_model, tmp_path):
        file_path = trained_model.save(tmp_path)
        loaded_model = PatchPCABaseline.load(file_path)
        assert loaded_model.n_components == trained_model.n_components