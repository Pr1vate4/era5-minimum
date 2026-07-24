import numpy as np
import pandas as pd
from .selection import get_split_timestamps


def month_to_season(month: int) -> str:
    if month in (12, 1, 2): return "DJF"
    if month in (3, 4, 5): return "MAM"
    if month in (6, 7, 8): return "JJA"
    return "SON"


def generate_nested_subsets(n_list: list[int] = [128, 256, 512, 1024, 2048, 4096, 8192], seed: int = 42) -> dict[
    int, list[str]]:
    train_times = get_split_timestamps("train")
    df = pd.DataFrame({"time": train_times, "season": train_times.month.map(month_to_season)})

    rng = np.random.default_rng(seed)
    max_n = max(n_list)

    # 1. Перемешиваем внутри каждого сезона
    season_pools = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        s_df = df[df["season"] == season].sample(frac=1, random_state=seed).reset_index(drop=True)
        season_pools[season] = s_df["time"].tolist()

    # 2. Round-Robin: берем по одному из каждого сезона по очереди
    interleaved = []
    pointers = {s: 0 for s in season_pools}
    while len(interleaved) < max_n:
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if len(interleaved) >= max_n: break
            if pointers[season] < len(season_pools[season]):
                interleaved.append(season_pools[season][pointers[season]])
                pointers[season] += 1

    # 3. Формируем вложенные словари (они уже сбалансированы по построению)
    subsets = {}
    for n in n_list:
        subset_times = interleaved[:n]
        subsets[n] = [t.strftime("%Y-%m-%dT%H:%M:%S") for t in subset_times]
    return subsets