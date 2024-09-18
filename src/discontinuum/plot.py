"""Plotting functions"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from discontinuum.engines.base import is_fitted
from scipy.stats import norm
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
    def plot_observations(self, ax: Optional[Axes] = None, **kwargs):
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
        # TODO provide easier access to data
        self.dm.data.target.plot.scatter(
            y=self.dm.data.target.name,
            c="k",
            s=5,
            linewidths=0.5,
            edgecolors="white",
            ax=ax,
            **kwargs,
        )

    @is_fitted
    def plot(self,
             covariates: Dataset,
             ci: float = 0.95,
             x: Optional[str] = None,
             ax: Optional[Axes] = None):
        """Plot predicted data.

        Parameters
        ----------
        covariates : Dataset
            Covariates.
        ci : float, optional
            Confidence interval. The default is 0.95.
        x : str, optional
            The coordinate to plot on the x axis.
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        if x is None:
            x = list(covariates.coords)[0]

        ax = self.setup_plot(ax)

        mu, se = self.predict(covariates, diag=True, pred_noise=True)

        target = DataArray(
            mu,
            coords=covariates.coords,
            dims=list(covariates.coords),
            attrs=self.dm.data.target.attrs,
        )

        # compute confidence bounds
        lower, upper = self.dm.error_pipeline.ci(target, se, ci=ci)

        target.plot.line(ax=ax, lw=1, zorder=2)

        ax.fill_between(
            target[x],
            lower,
            upper,
            color="b",
            alpha=0.1,
            zorder=1,
        )

        self.plot_observations(ax, zorder=3)

        return ax
