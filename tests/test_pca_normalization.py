import pytest
import numpy as np

from era5_minimum.baselines.normalization import(
    ChannelNormalizer,
    NormalizationStats,
    normalize,
    denormalize,
    _SAFE_STD_VALUE)


# ==============================================================================
# ЧАСТЬ 1: Fit нормализации (вычисление статистик)
# ==============================================================================
class TestPart1Fit:
    """Проверяется корректность вычисления mean, std и valid_count."""

    def test_mean_and_std_basic(self):
        """Проверка на простых данных с известным распределением."""
        # 2 канала, 3 изображения (time), 2x2 пространство
        # Канал 0: все значения = 2.0
        # Канал 1: значения [1, 2, 3, 4] (повторяются)
        data = np.zeros((3, 2, 2, 2))
        data[:, 0, :, :] = 2.0
        data[:, 1, 0, 0] = 1.0
        data[:, 1, 0, 1] = 2.0
        data[:, 1, 1, 0] = 3.0
        data[:, 1, 1, 1] = 4.0

        norm = ChannelNormalizer(["ch_a", "ch_b"])
        norm.update(data)
        stats = norm.finalize()

        # Канал 0: mean=2.0, std=0 (но сработает safe_std)
        assert np.isclose(stats.mean[0], 2.0)
        assert stats.used_safe_std[0] is True
        assert stats.std[0] == _SAFE_STD_VALUE

        # Канал 1: mean=2.5, std=sqrt(((1-2.5)^2 + ... + (4-2.5)^2)/4) = sqrt(1.25)
        assert np.isclose(stats.mean[1], 2.5)
        assert np.isclose(stats.std[1], np.sqrt(1.25))
        assert stats.used_safe_std[1] is False

    def test_different_channels_independence(self):
        """Статистики разных каналов должны вычисляться независимо."""
        data = np.random.rand(5, 3, 4, 4)
        # Задаем каналы с явно разными смещениями
        data[:, 0, :, :] += 10.0
        data[:, 1, :, :] += 20.0
        data[:, 2, :, :] += 30.0

        norm = ChannelNormalizer(["c1", "c2", "c3"])
        norm.update(data)
        stats = norm.finalize()

        assert stats.mean[0] < stats.mean[1] < stats.mean[2]
        assert np.isclose(stats.mean[0], 10.5, atol=0.5)
        assert np.isclose(stats.mean[1], 20.5, atol=0.5)
        assert np.isclose(stats.mean[2], 30.5, atol=0.5)

    def test_multiple_batches_streaming(self):
        """Потоковое обновление (несколько вызовов update) должно давать
        тот же результат, что и однократное обновление всем массивом."""
        np.random.seed(42)
        data_full = np.random.rand(10, 2, 8, 8)

        # Способ 1: Потоково (3 батча)
        norm_stream = ChannelNormalizer(["a", "b"])
        norm_stream.update(data_full[:4])
        norm_stream.update(data_full[4:7])
        norm_stream.update(data_full[7:])
        stats_stream = norm_stream.finalize()

        # Способ 2: Одним куском
        norm_batch = ChannelNormalizer(["a", "b"])
        norm_batch.update(data_full)
        stats_batch = norm_batch.finalize()

        np.testing.assert_allclose(stats_stream.mean, stats_batch.mean)
        np.testing.assert_allclose(stats_stream.std, stats_batch.std)
        np.testing.assert_array_equal(stats_stream.valid_count, stats_batch.valid_count)


# ==============================================================================
# ЧАСТЬ 2: Transform (применение нормализации)
# ==============================================================================
class TestPart2Transform:
    """Проверяется применение нормализации к данным."""

    @pytest.fixture
    def trained_stats_and_data(self):
        """Фикстура, которая возвращает и статистики, и данные, на которых они посчитаны."""
        np.random.seed(42)
        data = np.random.rand(5, 2, 10, 10) * 10.0 + 5.0
        norm = ChannelNormalizer(["c1", "c2"])
        norm.update(data)
        stats = norm.finalize()
        return stats, data

    def test_shape_preserved(self, trained_stats_and_data):
        """Форма массива после нормализации не должна изменяться."""
        stats, _ = trained_stats_and_data
        data = np.random.rand(3, 2, 10, 10)
        norm_data = normalize(data, stats)
        assert norm_data.shape == data.shape

    def test_values_normalized(self, trained_stats_and_data):
        """Нормализованные данные должны иметь mean~0 и std~1.
        ВАЖНО: проверять нужно на тех же данных, на которых считалась статистика!"""
        stats, data = trained_stats_and_data
        norm_data = normalize(data, stats)

        # Проверяем для каждого канала
        for c in range(2):
            assert np.abs(norm_data[:, c, :, :].mean()) < 1e-5, \
                f"Mean канала {c} не равен 0: {norm_data[:, c, :, :].mean()}"
            assert np.abs(norm_data[:, c, :, :].std() - 1.0) < 1e-5, \
                f"Std канала {c} не равен 1: {norm_data[:, c, :, :].std()}"

    def test_multiple_batches_and_channels(self, trained_stats_and_data):
        """Проверка работы на больших батчах и множественных каналах."""
        stats, _ = trained_stats_and_data
        data = np.random.rand(16, 2, 32, 32)
        norm_data = normalize(data, stats)

        # Убеждаемся, что нет NaN или Inf (если std не был 0)
        assert not np.any(np.isnan(norm_data))
        assert not np.any(np.isinf(norm_data))


# ==============================================================================
# ЧАСТЬ 3: Inverse Transform (Round-trip)
# ==============================================================================
class TestPart3InverseTransform:
    """Самый важный блок: проверка обратимости преобразований."""

    def test_round_trip_basic(self):
        """original -> normalize -> denormalize -> original."""
        np.random.seed(123)
        original = np.random.rand(4, 3, 16, 16) * 50.0 - 10.0

        norm = ChannelNormalizer(["x", "y", "z"])
        norm.update(original)
        stats = norm.finalize()

        transformed = normalize(original, stats)
        restored = denormalize(transformed, stats)

        np.testing.assert_allclose(restored, original, rtol=1e-5, atol=1e-6)

    def test_round_trip_with_safe_std(self):
        """Round-trip должен работать даже если std=0 (используется safe_std=1.0)."""
        # Канал 0: константа (std=0), Канал 1: случайные данные
        original = np.zeros((2, 2, 4, 4))
        original[:, 0, :, :] = 42.0
        original[:, 1, :, :] = np.random.rand(2, 4, 4)

        norm = ChannelNormalizer(["const", "rand"])
        norm.update(original)
        stats = norm.finalize()

        transformed = normalize(original, stats)
        restored = denormalize(transformed, stats)

        np.testing.assert_allclose(restored, original, rtol=1e-5, atol=1e-6)


# ==============================================================================
# ЧАСТЬ 4: Особые случаи (Требования ТЗ)
# ==============================================================================
class TestPart4SpecialCases:
    """Проверка устойчивости к NaN, маскам и отсутствия data leakage."""

    def test_nan_handling(self):
        """NaN не должны ломать обучение и должны исключаться из статистик."""
        data = np.array([
            [[[1.0, 2.0], [np.nan, 4.0]]],  # time 0
            [[[5.0, np.nan], [7.0, 8.0]]]  # time 1
        ])  # shape: [2, 1, 2, 2]

        # Валидные значения: 1, 2, 4, 5, 7, 8. Всего 6.
        # Mean = 27 / 6 = 4.5

        norm = ChannelNormalizer(["sst"])
        norm.update(data)
        stats = norm.finalize()

        assert stats.valid_count[0] == 6
        assert np.isclose(stats.mean[0], 4.5)
        # Дисперсия: sum((x - 4.5)^2) / 6
        expected_var = ((1 - 4.5) ** 2 + (2 - 4.5) ** 2 + (4 - 4.5) ** 2 + (5 - 4.5) ** 2 + (7 - 4.5) ** 2 + (
                    8 - 4.5) ** 2) / 6
        assert np.isclose(stats.std[0], np.sqrt(expected_var))

    def test_sst_mask_valid_mask(self):
        """Маска (например, суша для SST) должна корректно исключать точки."""
        # 1 канал, 1 изображение, 2x2
        data = np.array([[[[100.0, 2.0], [3.0, 4.0]]]])  # 100.0 - "суша" (выброс)

        # Маска: True - море, False - суша. Форма [time, height, width]
        mask = np.array([[[[False, True], [True, True]]]])

        norm = ChannelNormalizer(["sst"])
        norm.update(data, valid_mask=mask)
        stats = norm.finalize()

        # Должны посчитаться только 2, 3, 4. Mean = 3.0
        assert stats.valid_count[0] == 3
        assert np.isclose(stats.mean[0], 3.0)

    def test_no_data_leakage(self):
        """Статистики, посчитанные на train, не должны "подгоняться" под test.
        Если нормализовать test данными, используя train-статистики,
        распределение test НЕ должно стать N(0, 1)."""
        np.random.seed(99)
        train_data = np.random.rand(5, 2, 8, 8) * 2.0 + 10.0  # mean ~ 11
        test_data = np.random.rand(5, 2, 8, 8) * 2.0 + 50.0  # mean ~ 51 (сильно отличается)

        # Обучаем ТОЛЬКО на train
        norm = ChannelNormalizer(["c1", "c2"])
        norm.update(train_data)
        stats = norm.finalize()

        # Нормализуем test, используя train-статистики
        norm_test = normalize(test_data, stats)

        # Если бы был data leakage (или если бы мы ошибочно посчитали stats по test),
        # среднее было бы ~0. Но так как stats из train, среднее norm_test должно быть
        # примерно (51 - 11) / std ~ 40 / 1.5 > 10.
        assert np.abs(norm_test.mean()) > 10.0, "Data leakage detected! Test data influenced the stats."

    def test_safe_std_flag(self):
        """Проверка, что флаг used_safe_std корректно выставляется для константных каналов."""
        data = np.zeros((2, 2, 4, 4))
        data[:, 0, :, :] = 5.0  # Канал 0: константа
        data[:, 1, :, :] = np.random.rand(2, 4, 4)  # Канал 1: шум

        norm = ChannelNormalizer(["const", "noise"])
        norm.update(data)
        stats = norm.finalize()

        assert stats.used_safe_std[0] is True
        assert stats.std[0] == _SAFE_STD_VALUE
        assert stats.used_safe_std[1] is False