import pandas as pd

SPLITS = {
    "train": slice("2014-01-01T00:00:00", "2019-12-24T18:00:00"),
    "val": slice("2020-01-01T00:00:00", "2020-12-31T18:00:00"),
    "test": slice("2021-01-01T00:00:00", "2021-12-31T18:00:00"),
}


def get_split_timestamps(split: str) -> pd.DatetimeIndex:
    """Return authoritative six-hour UTC timestamps for a named split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(SPLITS)}")
    return pd.date_range(SPLITS[split].start, SPLITS[split].stop, freq="6h")
