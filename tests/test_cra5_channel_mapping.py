from __future__ import annotations

import pytest

from era5_minimum.data.channel_spec import CHANNEL_NAMES
from era5_minimum.cra5.channel_mapping import (
    CRA5_CHANNEL_MAPPING,
    CRA5_PRESSURE_LEVELS,
    CRA5_PRESSURE_VARIABLES,
    CRA5_SURFACE_VARIABLES,
    Cra5ChannelMapping,
    validate_cra5_channel_mapping,
)


def test_cra5_159_constants_describe_the_pinned_source_layout() -> None:
    assert CRA5_PRESSURE_VARIABLES == ("z", "q", "u", "v", "t", "w")
    assert CRA5_PRESSURE_LEVELS == (
        1000,
        950,
        925,
        900,
        850,
        800,
        700,
        600,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        70,
        50,
        30,
        20,
        10,
        7,
        5,
        3,
        2,
        1,
    )
    assert CRA5_SURFACE_VARIABLES == (
        "v10",
        "u10",
        "v100",
        "u100",
        "t2m",
        "tcc",
        "sp",
        "tp6h",
        "msl",
    )


def test_mapping_preserves_canonical_order_and_maps_only_supported_channels() -> None:
    assert tuple(row.canonical_name for row in CRA5_CHANNEL_MAPPING) == CHANNEL_NAMES
    assert all(isinstance(row, Cra5ChannelMapping) for row in CRA5_CHANNEL_MAPPING)
    assert sum(row.cra5_index is not None for row in CRA5_CHANNEL_MAPPING) == 26
    assert {
        row.canonical_name
        for row in CRA5_CHANNEL_MAPPING
        if row.initialization == "learned_boundary"
    } == {"sst", "tcwv"}
    assert {
        row.canonical_name
        for row in CRA5_CHANNEL_MAPPING
        if row.cra5_index is None
    } == {"sst", "tcwv"}


def test_mapping_uses_explicit_cra5_indices() -> None:
    by_name = {row.canonical_name: row for row in CRA5_CHANNEL_MAPPING}

    assert by_name["T1000"].cra5_name == "t1000"
    assert by_name["T1000"].cra5_index == 100
    assert by_name["U925"].cra5_name == "u925"
    assert by_name["U925"].cra5_index == 52
    assert by_name["Z700"].cra5_index == 6
    assert by_name["Q850"].cra5_index == 29
    assert by_name["v10"].cra5_index == 150
    assert by_name["u10"].cra5_index == 151
    assert by_name["t2m"].cra5_index == 154
    assert by_name["tcc"].cra5_index == 155
    assert by_name["tp6h"].cra5_index == 157
    assert by_name["mslp"].cra5_name == "msl"
    assert by_name["mslp"].cra5_index == 158
    assert by_name["sst"].initialization == "learned_boundary"
    assert by_name["tcwv"].initialization == "learned_boundary"


def test_mapping_validator_rejects_duplicate_source_indices() -> None:
    duplicate = list(CRA5_CHANNEL_MAPPING)
    duplicate[1] = Cra5ChannelMapping(
        canonical_name=duplicate[1].canonical_name,
        cra5_name=duplicate[1].cra5_name,
        cra5_index=duplicate[0].cra5_index,
        initialization=duplicate[1].initialization,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_cra5_channel_mapping(tuple(duplicate))
