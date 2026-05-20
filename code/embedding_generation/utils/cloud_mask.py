"""Sentinel-2 L2A cloud masking via the Scene Classification Layer (SCL)."""
import numpy as np
import xarray as xr


# SCL classes to exclude (set to NaN):
#   0  = No data
#   1  = Saturated / Defective
#   3  = Cloud Shadow
#   8  = Cloud (medium probability)
#   9  = Cloud (high probability)
#   10 = Thin Cirrus
# Optionally add 11 (Snow/Ice) if snow-free composites are needed.
CLOUDY_SCL = frozenset([0, 1, 3, 8, 9, 10])


def mask_s2_l2a(ds: xr.Dataset, scl_band: str = "SCL", exclude_snow: bool = False) -> xr.Dataset:
    """Return ds with cloudy/invalid pixels set to NaN, based on SCL.

    Args:
        ds: xarray Dataset from odc-stac with all bands + SCL.
        scl_band: Name of the SCL variable in ds.
        exclude_snow: If True, also mask SCL class 11 (Snow/Ice).
    """
    bad_classes = set(CLOUDY_SCL)
    if exclude_snow:
        bad_classes.add(11)

    scl = ds[scl_band]
    valid = ~scl.isin(list(bad_classes))  # True where pixel is valid

    band_names = [v for v in ds.data_vars if v != scl_band]
    masked = ds[band_names].where(valid)
    return masked
