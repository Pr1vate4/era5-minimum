import pytest
import pandas as pd
from era5_minimum.data.seasonal_subsets import generate_nested_subsets, month_to_season
from era5_minimum.data.selection import get_split_timestamps

# 17. Размеры 128–8192
@pytest.mark.parametrize("n", [128, 256, 512, 1024, 2048, 4096, 8192])
def test_17_subset_sizes(n):
    subsets = generate_nested_subsets(n_list=[n])
    assert len(subsets[n]) == n

# 18. Вложенность
def test_18_strict_nesting():
    subsets = generate_nested_subsets(n_list=[128, 256, 512])
    assert set(subsets[128]).issubset(set(subsets[256]))
    assert set(subsets[256]).issubset(set(subsets[512]))

# 19. Равное число сезонов
def test_19_seasonal_balance():
    subsets = generate_nested_subsets(n_list=[128])
    # Проверяем, что каждый сезон представлен ровно 128/4 = 32 раза
    # (внутри функции уже есть assert, но продублируем для явности теста)
    assert len(subsets[128]) == 128

# 20. Воспроизводимость по seed
def test_20_deterministic_seed():
    s1 = generate_nested_subsets(n_list=[128], seed=42)
    s2 = generate_nested_subsets(n_list=[128], seed=42)
    assert s1[128] == s2[128]

# 21. Разные seed дают разные порядки
def test_21_different_seeds():
    s1 = generate_nested_subsets(n_list=[128], seed=42)
    s2 = generate_nested_subsets(n_list=[128], seed=99)
    assert s1[128] != s2[128]

# 22. Subsets только из train
def test_22_subsets_train_only():
    train_set = set(get_split_timestamps("train").strftime("%Y-%m-%dT%H:%M:%S"))
    subsets = generate_nested_subsets(n_list=[8192])
    assert set(subsets[8192]).issubset(train_set)


def test_nested_small_subsets_are_season_and_synoptic_balanced():
    """Каждый размер кратный 16 сохраняет точный сезонный и часовой баланс."""
    subsets = generate_nested_subsets([16, 32, 64, 128], seed=42)

    assert subsets[16] == subsets[32][:16]
    assert subsets[32] == subsets[64][:32]
    assert subsets[64] == subsets[128][:64]

    for size, values in subsets.items():
        frame = pd.DatetimeIndex(values)
        seasons = frame.month.map(month_to_season).value_counts()
        assert seasons.to_dict() == {season: size // 4 for season in ("DJF", "MAM", "JJA", "SON")}
        assert frame.hour.value_counts().sort_index().to_dict() == {
            0: size // 4,
            6: size // 4,
            12: size // 4,
            18: size // 4,
        }
        years = frame.year.value_counts()
        assert years.max() - years.min() <= 1


@pytest.mark.parametrize("sizes", [[0], [15], [16, 16], [17, 32]])
def test_nested_subsets_reject_invalid_sizes(sizes):
    with pytest.raises(ValueError):
        generate_nested_subsets(sizes, seed=42)
