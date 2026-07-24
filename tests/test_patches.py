import pytest
import numpy as np

# Импорты из тестируемого модуля
from era5_minimum.baselines.patches import (
    PatchGridInfo,
    compute_padding,
    patch_grid_dims,
    extract_patches,
    reconstruct_from_patches,
)


# ==============================================================================
# ЧАСТЬ 1: Базовые функции, свойства и проверки ошибок
# ==============================================================================
class TestPart1Basics:
    """Тесты compute_padding, patch_grid_dims, свойств PatchGridInfo и ошибок."""

    def test_compute_padding_exact_multiple(self):
        assert compute_padding(10, 5) == 0
        assert compute_padding(12, 4) == 0
        assert compute_padding(7, 7) == 0

    def test_compute_padding_with_remainder(self):
        assert compute_padding(10, 3) == 2  # 10 -> 12
        assert compute_padding(10, 4) == 2  # 10 -> 12
        assert compute_padding(5, 3) == 1  # 5 -> 6

    def test_compute_padding_size_less_than_patch(self):
        # Если размер меньше патча, padding должен добить до размера патча
        assert compute_padding(2, 5) == 3
        assert compute_padding(1, 10) == 9

    def test_compute_padding_errors(self):
        with pytest.raises(ValueError, match="положительным"):
            compute_padding(10, 0)
        with pytest.raises(ValueError, match="положительным"):
            compute_padding(10, -5)

    def test_patch_grid_dims_exact(self):
        assert patch_grid_dims(10, 10, 5, 5) == (2, 2)
        assert patch_grid_dims(12, 8, 4, 4) == (3, 2)

    def test_patch_grid_dims_with_padding(self):
        # 10x10 с патчем 3x3 -> padding до 12x12 -> 4x4 патча
        assert patch_grid_dims(10, 10, 3, 3) == (4, 4)
        # 10x12 с патчем 3x4 -> pad_h=2, pad_w=0 -> 12x12 -> 4x3 патча
        assert patch_grid_dims(10, 12, 3, 4) == (4, 3)

    def test_patch_grid_info_properties(self):
        grid = PatchGridInfo(
            batch_size=2, channel_count=3,
            original_height=10, original_width=10,
            padded_height=12, padded_width=12,
            patch_height=4, patch_width=4,
            pad_bottom=2, pad_right=2,
            n_patches_h=3, n_patches_w=3
        )
        assert grid.patch_feature_count == 3 * 4 * 4  # 48
        assert grid.n_patches_per_sample == 3 * 3  # 9
        assert grid.n_patches_total == 2 * 9  # 18


# ==============================================================================
# ЧАСТЬ 2: extract_patches (разбиение на патчи)
# ==============================================================================
class TestPart2Extract:
    """Тесты extract_patches: размеры, padding, нечётные размеры, порядок, round-trip."""

    def test_extract_shapes_and_padding(self):
        # Batch=2, Channels=3, H=10, W=10. Patch=3x3.
        # Padding: H->12, W->12. Patches: 4x4=16 на сэмпл.
        data = np.zeros((2, 3, 10, 10))
        patches, grid = extract_patches(data, 3, 3)

        assert patches.shape == (2 * 16, 3 * 3 * 3)  # (32, 27)
        assert grid.padded_height == 12
        assert grid.padded_width == 12
        assert grid.pad_bottom == 2
        assert grid.pad_right == 2

    def test_extract_odd_sizes(self):
        # Нечётные размеры: 7x7, патч 3x3 -> padding до 9x9 -> 3x3 патча
        data = np.zeros((1, 1, 7, 7))
        patches, grid = extract_patches(data, 3, 3)

        assert patches.shape == (1 * 9, 1 * 3 * 3)  # (9, 9)
        assert grid.n_patches_h == 3
        assert grid.n_patches_w == 3

    def test_extract_patch_order(self):
        # Проверяем порядок: batch -> row -> col -> channel -> row -> col
        # Создаем массив, где значение пикселя = h * 10 + w
        data = np.zeros((1, 1, 4, 4))
        for h in range(4):
            for w in range(4):
                data[0, 0, h, w] = h * 10 + w

        patches, _ = extract_patches(data, 2, 2)

        # Патч 0 (верхний левый): h=0,1; w=0,1 -> 0, 1, 10, 11
        np.testing.assert_array_equal(patches[0], [0, 1, 10, 11])
        # Патч 1 (верхний правый): h=0,1; w=2,3 -> 2, 3, 12, 13
        np.testing.assert_array_equal(patches[1], [2, 3, 12, 13])
        # Патч 2 (нижний левый): h=2,3; w=0,1 -> 20, 21, 30, 31
        np.testing.assert_array_equal(patches[2], [20, 21, 30, 31])

    def test_extract_round_trip_basic(self):
        # Базовая проверка, что извлечение и сборка не теряют данные
        data = np.random.rand(1, 1, 10, 10)
        patches, grid = extract_patches(data, 3, 3)
        reconstructed = reconstruct_from_patches(patches, grid)
        np.testing.assert_array_almost_equal(data, reconstructed)


# ==============================================================================
# ЧАСТЬ 3: reconstruct_from_patches и финальные проверки ТЗ
# ==============================================================================
class TestPart3ReconstructAndFinal:
    """Тесты reconstruct_from_patches, исключения, многоканальность, батчи, ТЗ."""

    def test_reconstruct_exceptions_shape_mismatch(self):
        data = np.zeros((1, 1, 10, 10))
        _, grid = extract_patches(data, 5, 5)

        # Передаем патчи с неправильным количеством признаков
        wrong_features_patches = np.zeros((grid.n_patches_total, 10))
        with pytest.raises(ValueError, match="Форма patches не совпадает"):
            reconstruct_from_patches(wrong_features_patches, grid)

        # Передаем патчи с неправильным общим количеством
        wrong_total_patches = np.zeros((grid.n_patches_total + 1, grid.patch_feature_count))
        with pytest.raises(ValueError, match="Форма patches не совпадает"):
            reconstruct_from_patches(wrong_total_patches, grid)

    def test_reconstruct_multichannel(self):
        # 3 канала
        data = np.random.rand(1, 3, 10, 10)
        patches, grid = extract_patches(data, 4, 4)

        assert patches.shape == (1 * 9, 3 * 4 * 4)  # (9, 48)

        reconstructed = reconstruct_from_patches(patches, grid)
        np.testing.assert_array_almost_equal(data, reconstructed)

    def test_reconstruct_batch_gt_1(self):
        # Батч > 1 (например, 4 сэмпла)
        data = np.random.rand(4, 2, 15, 20)
        patches, grid = extract_patches(data, 5, 5)

        # 15x20 кратно 5x5, padding не нужен
        assert grid.pad_bottom == 0 and grid.pad_right == 0
        assert patches.shape == (4 * 3 * 4, 2 * 5 * 5)  # (48, 50)

        reconstructed = reconstruct_from_patches(patches, grid)
        np.testing.assert_array_almost_equal(data, reconstructed)

    def test_final_tz_requirements_check(self):
        """Финальная комплексная проверка требований из ТЗ (п. 4.2 и п. 6)."""
        # 1. Генерируем "сложные" данные (нечётные размеры, много каналов, большой батч)
        original_data = np.random.rand(3, 4, 17, 23)

        # 2. Извлекаем патчи
        patches, grid = extract_patches(original_data, 5, 7)

        # 3. Проверяем, что padding действительно zero-padding (п.6 ТЗ)
        # Для этого смотрим на промежуточный padded массив внутри grid (через reconstruct)
        # Мы не можем достать padded напрямую, но можем проверить, что обрезка работает корректно.

        # 4. Собираем обратно
        reconstructed_data = reconstruct_from_patches(patches, grid)

        # 5. ГЛАВНОЕ ТРЕБОВАНИЕ П.4.2: reconstructed.shape == original.shape
        assert reconstructed_data.shape == original_data.shape, \
            "Требование п.4.2 нарушено: формы оригинала и реконструкции не совпадают!"

        # 6. Данные не должны теряться или изменяться
        np.testing.assert_array_almost_equal(
            original_data, reconstructed_data,
            err_msg="Данные были потеряны или изменены при round-trip!"
        )

        # 7. Проверяем, что сумма элементов совпадает (доп. защита от "тихого" зануления)
        assert np.isclose(original_data.sum(), reconstructed_data.sum())