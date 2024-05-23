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


class LoadestPlotMixin:
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
        ax = self.setup_plot(ax)

        # TODO provide easier access to data
        self.dm.data.target.plot.scatter(
            y="concentration",
            c="k",
            s=5,
            linewidths=0.5,
            edgecolors="white",
            ax=ax,
            **kwargs,
        )

        return ax

    @is_fitted
    def plot(self, covariates: Dataset, ci: float = 0.95, ax: Optional[Axes] = None):
        """Plot predicted concentration versus time.

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
        alpha = (1 - ci)/2
        zscore = norm.ppf(1-alpha)

        cb = se**zscore

        target.plot.line(ax=ax, lw=1, zorder=2)

        ax.fill_between(
            target["time"], (target / cb), (target * cb), color="b", alpha=0.1, zorder=1
        )

        self.plot_observations(ax, zorder=3)

        return ax

    @is_fitted
    def plot_flux(self, covariates: Dataset, ax: Optional[Axes] = None):
        """Plot predicted flux versus time.

        Parameters
        ----------
        covariates : Dataset
            Covariates.

        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        # TODO FINISH
        ax = self.setup_plot(ax)
        mu, se = self.predict(covariates, diag=True, pred_noise=True)

        flux = self.daily_flux(mu, covariates["flow"])
        flux_se = self.daily_flux(se, covariates["flow"])

        ci = 1.96 * flux_se

        flux.plot.line(ax=ax, lw=1, zorder=2)

        ax.fill_between(
            flux["time"], (flux - ci), (flux + ci), color="b", alpha=0.1, zorder=1
        )

        return ax

    @is_fitted
    def observed_vs_predicted(self, ax: Optional[Axes] = None):
        """Plot observed versus predicted concentrations.

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

        return ax

    @is_fitted
    def contourf(
        self,
        covariate: str = "flow",
        ax: Optional[Axes] = None,
        cbar_kwargs: Optional[Dict] = None,
        y_scale: str = "log",
        **kwargs,
    ):
        """Plot contourf

        TODO: plot any pair of variables on x and y axes.

        Parameters
        ----------
        covariate : str
            Covariate to plot.
        ax : Axes, optional
            Pre-defined matplotlib axes.
        cbar_kwargs : dict, optional
            Colorbar keyword arguments. The default is None.
        y_scale : str, optional
            y-axis scale. The default is "log".
        kwargs : dict
            Contourf keyword arguments.


        """
        ax = self.setup_plot(ax)

        if cbar_kwargs is None:
            cbar_kwargs = {}

        cbar_defaults = {"rotation": 270, "labelpad": 25}
        cbar_defaults.update(cbar_kwargs)

        ax = self.setup_plot(ax)
        ax.set_yscale(y_scale)

        target, time, cov = self.predict_grid(covariate=covariate, t_step=12)
        X2, X1 = np.meshgrid(cov, time)  # might need to flip these
        cs = ax.contourf(X1, X2, target, **kwargs)

        ax.set_ylabel(label_from_attrs(self.dm.data.covariates[covariate]))
        ax.set_xlabel("Year")

        fig = ax.get_figure()
        cbar = fig.colorbar(cs, ax=ax)
        cbar.ax.set_ylabel(label_from_attrs(self.dm.data.target), **cbar_defaults)
        return ax

    @is_fitted
    def countourf_diff(self, ax: Optional[Axes] = None):
        pass
