from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from xarray import DataArray


def compute_flux(self, concentration, flow):
    """Compute load from concentration and flow.

    Assumes data are at daily resolution.

    Parameters
    ----------
    concentration : DataArray
        Concentration in mg/L.
    flow : DataArray
        Flow in cubic meters per second.

    Returns
    -------
    DataArray
        Flux in kg per day.
    """
    attrs = concentration.attrs.copy()

    attrs.update({"units": "kg per day"})
    SECONDS_TO_DAY = 86400
    MG_TO_KG = 1e-6
    L_TO_M3 = 1e-3
    # TODO double check conversion
    a = concentration * MG_TO_KG * flow * SECONDS_TO_DAY * L_TO_M3
    da = DataArray(a, dims=["time"], coords=[flow.time], attrs=attrs)
    return da
