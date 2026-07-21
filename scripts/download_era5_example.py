"""Download a tiny raw ERA5 single-level sample through CDS API.

Before running:
1. Create a CDS account and accept the dataset licence.
2. Put the personal token in ~/.cdsapirc.
3. Install cdsapi.

Important: raw `total_precipitation` is NOT automatically equivalent to the
case variable `tp6h`. Confirm the supplied dataset semantics with organizers
before implementing the 6-hour aggregation.
"""
from __future__ import annotations

from pathlib import Path

import cdsapi


def main() -> None:
    target = Path("data/raw/era5_single_2024_01_01.nc")
    target.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature",
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_precipitation",
                "sea_surface_temperature",
                "total_column_water_vapour",
                "total_cloud_cover",
            ],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "grid": [0.5, 0.5],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    print(target)


if __name__ == "__main__":
    main()
