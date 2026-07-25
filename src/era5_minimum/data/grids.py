import numpy as np
import xarray as xr


def get_target_grid_05() -> xr.Dataset:
    """Return the canonical 0.5° cell-centred global target grid.

    The coordinates are deliberately exposed as coordinates named
    ``latitude`` and ``longitude``.  A data variable called ``lat``/``lon``
    is not enough for xarray remappers and had previously made the target
    grid ambiguous to consumers.
    """

    return xr.Dataset(
        coords={
            "latitude": np.linspace(-89.75, 89.75, 360, dtype=np.float64),
            "longitude": np.linspace(0.25, 359.75, 720, dtype=np.float64),
        }
    )
