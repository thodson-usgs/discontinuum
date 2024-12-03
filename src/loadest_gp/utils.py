from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from typing import Optional

    from matplotlib.pyplot import Axes
    from xarray import DataArray


def concentration_to_flux(
        concentration: DataArray,
        flow: DataArray,
        ) -> DataArray:
    """Convert concentration (mg/l) to flux (kg).

    Parameters
    ----------
    concentration : DataArray
        Concentration data.
    flow : DataArray
        Flow data.

    Returns
    -------
    DataArray
        Flux data.
    """
    time_delta = np.unique(
        concentration.time.diff(dim="time").dt.total_seconds()
    )
    mg_l_to_kg_m3 = 1e-3

    if len(time_delta) != 1:
        warn("Time delta is not constant",
             stacklevel=UserWarning,)

    if flow.units != "cubic meters per second":
        warn(
            "Check that flow is 'cubic meters per second'.",
            "Set flow.units = 'cubic meters per second' to silence.",
            stacklevel=UserWarning,
            )

    if "mg/l" not in concentration.units:
        warn("Check that concentration is in 'mg/l'.",
             "Set concentration.units = 'mg/l' to silence.",
             stacklevel=UserWarning,
             )

    flux = concentration * flow * time_delta * mg_l_to_kg_m3
    flux.attrs = concentration.attrs
    flux.attrs["units"] = "kilograms"
    flux.attrs["standard_name"] = "flux"
    return flux


def plot_annual_flux(
        flux: DataArray,
        ax: Optional[Axes] = None,
        **boxplot_kwargs,
        ) -> Axes:
    """Plot annual flux.

    Parameters
    ----------
    flux : DataArray
        Flux data.
    ax : Axes, optional
        Pre-defined matplotlib axes.
    **boxplot_kwargs
        Additional keyword arguments for matplotlib boxplot.

    Returns
    -------
    ax : Axes
        Generated matplotlib axes.
    """
    default_boxplot_kwargs = {
        "showfliers": False,
        "grid": False,
        "showcaps": False,
    }
    default_boxplot_kwargs.update(boxplot_kwargs)

    _, ax = plt.subplots() if ax is None else (ax.figure, ax)

    annual = flux.resample(time="YE").sum()
    annual.attrs["units"] = "kilograms per year"

    annual_df = annual.to_dataframe(name=annual.attrs["standard_name"])
    annual_df.boxplot(by="time", ax=ax, **default_boxplot_kwargs)

    ax.set_ylabel(
        "{}\n[{}]".format(annual.attrs["long_name"], annual.attrs["units"])
        )
    ax.set_xlabel("Year")
    ax.tick_params(axis="x", labelrotation=90)

    years = annual.time.dt.year.values
    labels = ["" if (year % 5 != 0) else year for year in years]
    ax.set_xticklabels(labels)
    ax.set_title("")
    return ax
