import json

import numpy as np
import xarray as xr

from era5_minimum.data.inspect_netcdf import write_report


def test_inspector_reports_tiny_netcdf_schema(tmp_path) -> None:
    times = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00"],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        data_vars={
            "2m_temperature": (
                ("time", "latitude", "longitude"),
                np.array([[[280.0, np.nan], [281.0, 282.0]]] * 3, dtype=np.float32),
                {"units": "K", "long_name": "2 metre temperature"},
            ),
            "mean_sea_level_pressure": (
                ("time", "latitude", "longitude"),
                np.full((3, 2, 2), 101325.0, dtype=np.float32),
                {"units": "Pa", "long_name": "Mean sea level pressure"},
            ),
        },
        coords={
            "time": times,
            "latitude": np.array([90.0, 0.0]),
            "longitude": np.array([0.0, 180.0]),
        },
    )
    input_path = tmp_path / "tiny_era5.nc"
    output_path = tmp_path / "schema_report.json"
    dataset.to_netcdf(input_path)

    report = write_report(input_path, output_path)
    saved_report = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.is_file()
    assert saved_report == report
    assert report["dimensions"] == {"time": 3, "latitude": 2, "longitude": 2}
    assert report["time"]["minimum"] == "2024-01-01T00:00:00Z"
    assert report["time"]["maximum"] == "2024-01-01T12:00:00Z"
    assert report["time"]["cadence"] == {"seconds": 21600.0, "uniform": True}
    assert report["latitude"]["direction"] == "descending"
    assert report["variables"]["2m_temperature"]["attributes"]["units"] == "K"
    assert report["variables"]["2m_temperature"]["statistics"]["nan_count"] == 3
