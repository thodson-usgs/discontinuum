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

        # This converts the input uncertainty from GSE to SE
        se_unc = np.log(self.dm.data.target_unc) * self.dm.data.target
        ax.errorbar(self.dm.data.covariates["stage"],
                    self.dm.data.target,
                    yerr=se_unc,
                    c="k",
                    marker='',
                    linestyle='',
                    capsize=2.5)
        data.plot.scatter(x='stage',
                          y='discharge',
                          hue="time",
                          add_colorbar=False,
                          add_legend=False,
                          marker='o',
                          edgecolors='face',
                          s=10,
                          ax=ax,
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
            x='time',
            y='stage',
            hue='time',
            s=10,
            edgecolors="face",
            ax=ax,
            zorder=2,
            add_legend=False,
            add_colorbar=False,
            **kwargs,
        )

        if covariates is not None:
            covariates["stage"].plot.line(ax=ax, lw=1, zorder=1)

        return ax

    @is_fitted
    def plot_discharge(self,
                       covariates: Dataset = None,
                       ci: float = 0.95,
                       ax: Optional[Axes] = None
                      ):
        """Plot predicted discharge versus time.

        Parameters
        ----------
        covariates : Dataset, optional
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

        if covariates is not None:
            mu, se = self.predict(covariates, diag=True, pred_noise=True)
    
            target = DataArray(
                mu,
                coords=[covariates.time],
                dims=["time"],
                attrs=self.dm.data.target.attrs,
            )
    
            # compute confidence bounds
            lower, upper = self.dm.error_pipeline.ci(target, se, ci=ci)

            target.plot.line(ax=ax, lw=1, zorder=2)

            ax.fill_between(
                target["time"],
                lower,
                upper,
                color="b",
                alpha=0.1,
                zorder=1,
            )

        self.dm.data.target.plot.scatter(
            x='time',
            y='discharge',
            hue='time',
            s=10,
            edgecolors="face",
            ax=ax,
            zorder=2,
            add_legend=False,
            add_colorbar=False,
        )

        return ax
