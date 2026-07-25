from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .selection import get_split_timestamps

SEASONS = ("DJF", "MAM", "JJA", "SON")
UTC_HOURS = (0, 6, 12, 18)
TRAIN_YEARS = (2014, 2015, 2016, 2017, 2018, 2019)
BLOCK_SIZE = len(SEASONS) * len(UTC_HOURS)
DEFAULT_SUBSET_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)


def month_to_season(month: int) -> str:
    """Map a Gregorian month number to a meteorological season."""
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(f"Invalid Gregorian month: {month}")


def _matching_cells_to_years(
    cells: list[tuple[str, int]],
    year_slots: list[int],
    pools: dict[tuple[str, int, int], list[pd.Timestamp]],
    rng: np.random.Generator,
) -> dict[tuple[str, int], int]:
    """Match each season/hour cell to one available year slot."""
    cell_order = list(cells)
    rng.shuffle(cell_order)
    adjacency = {
        cell: [
            slot
            for slot, year in enumerate(year_slots)
            if pools[(cell[0], cell[1], year)]
        ]
        for cell in cell_order
    }
    ordered_cells = sorted(cell_order, key=lambda cell: len(adjacency[cell]))
    matched_slots: dict[int, tuple[str, int]] = {}

    def visit(cell: tuple[str, int], visited: set[int]) -> bool:
        for slot in adjacency[cell]:
            if slot in visited:
                continue
            visited.add(slot)
            previous = matched_slots.get(slot)
            if previous is None or visit(previous, visited):
                matched_slots[slot] = cell
                return True
        return False

    if not all(visit(cell, set()) for cell in ordered_cells):
        raise ValueError("Training pool cannot satisfy balanced season/hour/year selection")
    return {cell: year_slots[slot] for slot, cell in matched_slots.items()}


def generate_nested_subsets(
    n_list: Sequence[int] = DEFAULT_SUBSET_SIZES,
    seed: int = 42,
) -> dict[int, list[str]]:
    """Generate deterministic nested, seasonally and synoptically balanced samples.

    Every requested size must be a distinct positive multiple of sixteen. Each
    sixteen-frame block contains one timestamp for every season/hour pair, and
    the selected training years stay balanced to within one frame.
    """
    train_times = get_split_timestamps("train")
    sizes = list(n_list)
    if not sizes:
        raise ValueError("At least one subset size is required")
    if len(set(sizes)) != len(sizes):
        raise ValueError("Subset sizes must be unique")
    if any(size <= 0 or size % BLOCK_SIZE != 0 for size in sizes):
        raise ValueError(f"Subset sizes must be positive multiples of {BLOCK_SIZE}")
    max_n = max(sizes)
    if max_n > len(train_times):
        raise ValueError(f"Requested {max_n} timestamps, but train pool has {len(train_times)}")

    rng = np.random.default_rng(seed)
    cells = [(season, hour) for season in SEASONS for hour in UTC_HOURS]
    pools: dict[tuple[str, int, int], list[pd.Timestamp]] = {
        (season, hour, year): []
        for season in SEASONS
        for hour in UTC_HOURS
        for year in TRAIN_YEARS
    }
    for timestamp in train_times:
        key = (month_to_season(timestamp.month), timestamp.hour, timestamp.year)
        pools[key].append(timestamp)
    for candidates in pools.values():
        rng.shuffle(candidates)

    selected: list[pd.Timestamp] = []
    for block in range(max_n // BLOCK_SIZE):
        year_slots = [TRAIN_YEARS[(block * BLOCK_SIZE + offset) % len(TRAIN_YEARS)] for offset in range(BLOCK_SIZE)]
        assignments = _matching_cells_to_years(cells, year_slots, pools, rng)
        block_values: list[pd.Timestamp] = []
        for cell in cells:
            year = assignments[cell]
            block_values.append(pools[(cell[0], cell[1], year)].pop())
        selected.extend(block_values)

    timestamp_strings = [timestamp.strftime("%Y-%m-%dT%H:%M:%S") for timestamp in selected]
    return {size: timestamp_strings[:size] for size in sizes}
