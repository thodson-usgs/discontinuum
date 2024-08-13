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
    def plot_observed_rating(self, ax: Optional[Axes] = None, **kwargs):
        """Plot stage observations versus discharge observations.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        ax = self.setup_plot(ax)

        data = xr.merge([self.dm.data.covariates["stage"],
                         self.dm.data.target])
        data.plot.scatter(x='stage',
                          y='discharge',
                          c="k",
                          marker='o',
                          s=20,
                          ax=ax,
                          **kwargs)

        # This converts the input uncertainty from GSE to SE
        se_unc = np.log(self.dm.data.target_unc) * self.dm.data.target
        ax.errorbar(self.dm.data.covariates["stage"],
                    self.dm.data.target,
                    yerr=se_unc,
                    c="k",
                    marker='',
                    linestyle='',
                    capsize=2.5,
                    **kwargs)

        return ax

    @is_fitted
    def plot_rating(self,
                    covariates: Dataset,
                    ci: float = 0.95,
                    ax: Optional[Axes] = None):
        """Plot predicted discharge versus stage.

        Parameters
        ----------
        covariates : Dataset
            Covariates.
        ci : float, optional
            Confidence interval. The default is 0.95.
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        ax = self.setup_plot(ax)

        mu, se = self.predict(covariates, diag=True, pred_noise=True)

        target = DataArray(
            mu,
            coords=[covariates.time],
            dims=["time"],
            attrs=self.dm.data.target.attrs,
        )

        # compute confidence bounds
        lower, upper = self.dm.error_pipeline.ci(target, se, ci=ci)

        ax.plot(covariates["stage"], target, lw=1, zorder=2)

        ax.fill_between(
            covariates['stage'],
            lower,
            upper,
            color="b",
            alpha=0.1,
            zorder=1,
        )

        self.plot_observed_rating(ax, zorder=3)

        return ax

    @is_fitted
    def plot_stage(self,
                   covariates: Dataset = None,
                   ax: Optional[Axes] = None,
                   **kwargs):
        """Plot observations versus time.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.

        """
        self.dm.data.covariates.plot.scatter(
            y='stage',
            c="k",
            s=5,
            linewidths=0.5,
            edgecolors="white",
            ax=ax,
            zorder=2,
            **kwargs,
        )

        if covariates is not None:
            covariates["stage"].plot.line(ax=ax, lw=1, zorder=1)

        return ax
