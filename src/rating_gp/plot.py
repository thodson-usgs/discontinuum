"""Plotting functions"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from discontinuum.engines.base import is_fitted
from scipy.stats import norm
import xarray as xr
from xarray import DataArray
from xarray.plot.utils import label_from_attrs

if TYPE_CHECKING:
    from typing import Dict, Optional

    from matplotlib.pyplot import Axes
    from xarray import Dataset


DEFAULT_FIGSIZE = (5, 5)
NARROW_LINE = 1
REGULAR_LINE = NARROW_LINE * 1.5


class RatingPlotMixin:
    """Mixin plotting functions for Model class"""

    @staticmethod
    def setup_plot(ax: Optional[Axes] = None):
        """Sets up figure and axes for rating curve plot.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        if ax is None:
            _, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

        return ax

    @is_fitted
    def plot_observations(self, ax: Optional[Axes] = None, **kwargs):
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
        # unc = np.abs(np.log(self.dm.data.target_unc) * self.dm.data.target)
        ax.errorbar(self.dm.data.covariates["stage"],
                    self.dm.data.target,
                    yerr=self.dm.data.target_unc,
                    # yerr=unc,
                    c="k",
                    marker='',
                    linestyle='',
                    capsize=2.5,
                    **kwargs)

        return ax

    @is_fitted
    def plot(self, covariates: Dataset, ci: float = 0.95, ax: Optional[Axes] = None):
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

        mu, se = self.predict(covariates.sortby('stage'),
                              diag=True,
                              pred_noise=True)

        target = DataArray(
            mu,
            coords=[covariates.time],
            dims=["time"],
            attrs=self.dm.data.target.attrs,
        )
        alpha = (1 - ci)/2
        zscore = norm.ppf(1-alpha)

        ax.plot(covariates["stage"], target, lw=1, zorder=2)

        if self.model_config.transform == "log":
            cb = se**zscore
            ax.fill_between(
                covariates['stage'],
                target / cb,
                target * cb,
                color="b",
                alpha=0.1,
                zorder=1,
            )

        elif self.model_config.transform == "standard":
            cb = se * zscore
            ax.fill_between(
                covariates['stage'],
                target - cb,
                target + cb,
                color="b",
                alpha=0.1,
                zorder=1,
            )


        self.plot_observations(ax, zorder=3)

        return ax

    @is_fitted
    def plot_observed_timeseries(self,
                                 variable: Optional[str] = "stage",
                                 ax: Optional[Axes] = None,
                                 **kwargs
                                ):
        """Plot an observed variable versus time.

        Parameters
        ----------
        variable : str, optional
            Variable whose time series is to be plotted.
            Options are 'stage' or 'discharge'.
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        ax = self.setup_plot(ax)

        data = xr.merge([self.dm.data.covariates,
                         self.dm.data.target])
        data.plot.scatter(
            y=variable,
            c="k",
            s=5,
            linewidths=0.5,
            edgecolors="white",
            ax=ax,
            **kwargs,
        )

        return ax

    @is_fitted
    def plot_timeseries(self,
                        covariates: Dataset,
                        variable: Optional[str] = "stage",
                        ci: float = 0.95,
                        ax: Optional[Axes] = None,
                        **kwargs
                       ):
        """Plot an observed variable versus time.

        Parameters
        ----------
        covariates : Dataset
            Covariates.
        variable : str, optional
            Variable whose time series is to be plotted.
            Options are 'stage' or 'discharge'.
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

        if variable == 'stage':
            covariates["stage"].plot(lw=1, zorder=2)
        elif variable == 'discharge':
            mu, se = self.predict(covariates, diag=True, pred_noise=True)

            target = DataArray(
                mu,
                coords=[covariates.time],
                dims=["time"],
                attrs=self.dm.data.target.attrs,
            )
            alpha = (1 - ci)/2
            zscore = norm.ppf(1-alpha)

            target.plot(lw=1, zorder=2)

            if self.model_config.transform == "log":
                cb = se**zscore
                ax.fill_between(
                    target['time'],
                    target / cb,
                    target * cb,
                    color="b",
                    alpha=0.1,
                    zorder=1,
                )
    
            elif self.model_config.transform == "standard":
                cb = se * zscore
                ax.fill_between(
                    target['time'],
                    target - cb,
                    target + cb,
                    color="b",
                    alpha=0.1,
                    zorder=1,
                )

        self.plot_observed_timeseries(variable=variable, ax=ax, zorder=3)

        return ax
