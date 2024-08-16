"""Plotting functions"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from discontinuum.engines.base import is_fitted
from discontinuum.plot import BasePlotMixin
from scipy.stats import norm
import xarray as xr
from xarray import DataArray
from xarray.plot.utils import label_from_attrs

if TYPE_CHECKING:
    from typing import Dict, Optional

    from matplotlib.pyplot import Axes
    from xarray import Dataset


DEFAULT_FIGSIZE = (5, 5)


class RatingPlotMixin(BasePlotMixin):
    """Mixin plotting functions for Model class"""

    @is_fitted
    def plot_rating_in_time(self,
                            time: str,
                            ci: float = 0.95,
                            ax: Optional[Axes] = None):
        """Plot predicted discharge versus stage at a given point in time.

        Parameters
        ----------
        time : str
            A date in the form of YYYY-MM-DD which to plot the rating.
        ci : float, optional
            Confidence interval. The default is 0.95.
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        n = 250
        stage = np.linspace(self.dm.data.covariates['stage'].min(),
                            self.dm.data.covariates['stage'].max(),
                            n)
        time = np.repeat(np.datetime64(f"{time} 00:00:00", 'ns'), n)
        
        ds = xr.Dataset(
            data_vars=dict(
                stage=(["time"], stage),
            ),
            coords=dict(
                time=time,
            ),
        )

        self.plot_rating(ds)

        return ax
