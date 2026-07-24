import pandas as pd

SPLITS = {
    "train": slice("2014-01-01", "2019-12-31"),
    "val": slice("2020-01-01", "2020-12-31"),
    "test": slice("2021-01-01", "2021-12-31"),
}

def get_split_timestamps(split: str) -> pd.DatetimeIndex:
    """Возвращает timestamps для заданного сплита (6-часовой шаг)."""
    assert split in SPLITS
    # Генерируем 6-часовые шаги для диапазона
    return pd.date_range(SPLITS[split].start, SPLITS[split].stop, freq="6h")