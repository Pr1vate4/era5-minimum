"""Offline checks for deterministic WeatherBench2 pilot timestamp selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_weatherbench2_05.py"
_SPEC = importlib.util.spec_from_file_location("prepare_weatherbench2_05", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_evaluation_selection_is_balanced_and_deterministic() -> None:
    first = _MODULE._evaluation_timestamps("validation", 16, 43)
    second = _MODULE._evaluation_timestamps("validation", 16, 43)
    assert first == second
    assert len(first) == len(set(first)) == 16
    assert {value[11:13] for value in first} == {"00", "06", "12", "18"}


def test_training_shard_is_exact_slice_of_n128_subset() -> None:
    full = _MODULE.generate_nested_subsets([128], seed=42)[128]
    shard = _MODULE._training_timestamps(population_size=128, start_index=16, count=56, seed=42)
    assert shard == full[16:72]
    assert len(shard) == 56


def test_empty_evaluation_split_is_allowed_for_a_train_shard() -> None:
    assert _MODULE._evaluation_timestamps("validation", 0, 43) == []
