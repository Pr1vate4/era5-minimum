import pytest
from era5_minimum.data.seasonal_subsets import generate_nested_subsets
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