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


class BasePlotMixin:
    """Mixin plotting functions for Model class"""

    @staticmethod
    def setup_plot(ax: Optional[Axes] = None):
        """Sets up figure and axes for plot.

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
    def plot_observations(self,
                          xdim: str = 'time',
                          ydim: str = None,
                          ax: Optional[Axes] = None,
                          **kwargs):
        """Plot observations.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.

        """
        if ydim is None:
            ydim = self.dm.data.target.name

        ax = self.setup_plot(ax)
        
        data = xr.merge([self.dm.data.covariates,
                         self.dm.data.target])
        # This converts the input uncertainty from GSE to SE
        if self.dm.data.target_unc is not None:
            if xdim == self.dm.data.target.name:
                xerr = np.log(self.dm.data.target_unc) * self.dm.data.target
                yerr = None
            if ydim == self.dm.data.target.name:
                yerr = np.log(self.dm.data.target_unc) * self.dm.data.target
                xerr = None
        else:
            xerr = None
            yerr = None

        ax.errorbar(data[xdim].values,
                    data[ydim].values,
                    xerr=xerr,
                    yerr=yerr,
                    c="k",
                    marker='',
                    linestyle='',
                    capsize=2.5,
                    **kwargs)

        data.plot.scatter(
            x=xdim,
            y=ydim,
            c="k",
            s=5,
            linewidths=0.5,
            edgecolors="white",
            ax=ax,
            **kwargs,
        )

        return ax


    @is_fitted
    def plot_predictions(self,
                         covariates: Dataset,
                         xdim: str = 'time',
                         ydim: str = None,
                         ci: float = 0.95,
                         ax: Optional[Axes] = None,
                         **kwargs):
        """Plot predicted values.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.

        """
        if ydim is None:
            ydim = self.dm.data.target.name

        if xdim == self.dm.data.target.name:
            covariates = covariates.sortby(ydim)
        elif ydim == self.dm.data.target.name:
            covariates = covariates.sortby(xdim)

        mu, se = self.predict(covariates, diag=True, pred_noise=True)

        # compute confidence bounds
        lower, upper = self.dm.error_pipeline.ci(mu, se, ci=ci)
        data = xr.merge([covariates,
                         mu,
                         lower.rename(f'lower'),
                         upper.rename(f'upper')])

        ax = self.setup_plot(ax)

        ax.plot(data[xdim], data[ydim], lw=1, zorder=2)
        ax.set_xlabel(label_from_attrs(data[xdim]))
        ax.set_ylabel(label_from_attrs(data[ydim]))

        if ydim == self.dm.data.target.name:
            ax.fill_between(
                data[xdim],
                data['lower'],
                data['upper'],
                color="b",
                alpha=0.1,
                zorder=1,
            )
        elif xdim == self.dm.data.target.name:
            ax.fill_betweenx(
                data[ydim],
                data['lower'],
                data['upper'],
                color="b",
                alpha=0.1,
                zorder=1,
            )

        return ax
            

    @is_fitted
    def plot(self,
             covariates: Dataset,
             xdim: str = 'time',
             ydim: str = None,
             ci: float = 0.95,
             ax: Optional[Axes] = None):
        """Plot predicted and observed values.

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

        self.plot_observations(ax=ax, xdim=xdim, ydim=ydim, zorder=3)
        self.plot_predictions(covariates, xdim=xdim, ydim=ydim, ax=ax, zorder=3)

        return ax
