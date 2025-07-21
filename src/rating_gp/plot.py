"""Plotting functions"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from discontinuum.engines.base import is_fitted
from discontinuum.plot import BasePlotMixin
from scipy.stats import norm
from rating_gp.models.kernels import SigmoidKernel
import xarray as xr
from xarray import DataArray
from xarray.plot.utils import label_from_attrs

if TYPE_CHECKING:
    from typing import Dict, Optional, Union
    from datetime import datetime
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
                    ax: Optional[Axes] = None,
                    **kwargs
                   ):
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

        ax.plot(covariates["stage"], target, lw=1, zorder=2, **kwargs)

        ax.fill_between(
            covariates['stage'],
            lower,
            upper,
            alpha=0.1,
            zorder=1,
            **kwargs
        )
        # self.plot_observed_rating(ax, zorder=3)

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
        ax = self.setup_plot(ax)

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

    @is_fitted
    def add_time_colorbar(self,
                          ax: Optional[Axes] = None,
                          time_ticks: Optional[list[Union[str, datetime]]] = None,
                          **kwargs
                         ):
        """Add a colorbar of the data time range to the figure.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        time_ticks : list[str, datetime], optional
            List of times to place ticks on the colorbar.

        Returns
        -------
        cbar : Colorbar
            Generated matplotlib colorbar.
        """
        cbar_range = pd.to_datetime(
            [self.dm.data.covariates.time.min().values,
             self.dm.data.covariates.time.max().values]
        )
        if time_ticks is None:
            time_ticks = pd.date_range(*cbar_range, periods=5)

        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
        else:
            cmap = None

        ticks = (time_ticks - cbar_range[0])/(cbar_range[1] - cbar_range[0])
        if not hasattr(ax, 'cbar'):
            cbar = plt.colorbar(ScalarMappable(cmap=cmap),
                                ax=ax,
                                aspect=30,
                                ticks=ticks,
                                format=mticker.FixedFormatter(
                                    time_ticks.strftime('%Y-%m-%d')
                                ),
                                **kwargs
                               )
        else:
            cbar = ax.cbar

        ax.cbar = cbar

        return cbar


    @is_fitted
    def plot_ratings_in_time(self,
                             time: Optional[list[Union[str, datetime]]] = None,
                             ci: Optional[int] = 0,
                             ax: Optional[Axes] = None,
                             **kwargs):
        """Plot predicted discharge versus stage.

        Parameters
        ----------
        time : list[str, datetime], optional
            List of times at which to plot the rating curve.
        ci : float, optional
            Confidence interval. Default is 0 (i.e., no confidence intervals).
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        ax = self.setup_plot(ax)
        
        n = 250
        stage = np.linspace(self.dm.data.covariates['stage'].min().values*0.95,
                            self.dm.data.covariates['stage'].max().values*1.5,
                            n)
        
        if time is None:
            time = pd.date_range(
                self.dm.data.covariates['time'].min().values,
                self.dm.data.covariates['time'].max().values,
                periods=5
            )
        else:
            time = pd.to_datetime(time)
        
        cbar = self.add_time_colorbar(ax=ax, time_ticks=time, **kwargs)
        cbar_range = pd.to_datetime(
            [self.dm.data.covariates.time.min().values,
             self.dm.data.covariates.time.max().values]
        )

        for datetime in time:
            covariates = xr.Dataset(
                data_vars=dict(
                    stage=(["time"], stage),
                ),
                coords=dict(
                    time=np.repeat(datetime, n),
                ),
            )
            self.plot_rating(covariates,
                             ci=ci,
                             ax=ax,
                             color=cbar.cmap(
                                 ((datetime - cbar_range[0])
                                  /(cbar_range[1] - cbar_range[0]))
                             )
                             )

        return ax
