"""Plotting functions"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from xarray import DataArray
from scipy.stats import norm

from discontinuum.engines.base import is_fitted

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes


DEFAULT_FIGSIZE = (5, 5)
NARROW_LINE = 1
REGULAR_LINE = NARROW_LINE * 1.5


class PlotMixin:
    """Mixin plotting functions for Model class"""

    @staticmethod
    def setup_plot(ax: Axes = None):
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
    def plot_observations(self, ax: Axes = None, **kwargs):
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
            y='concentration',
            c='k',
            s=5,
            linewidths=0.5,
            edgecolors='white',
            ax=ax,
            **kwargs,
        )

        return ax

    @is_fitted
    def plot_concentration(self, covariates, ci=0.95, ax: Axes = None):
        """Plot predicted concentration versus time.

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

        mu, se = self.predict(covariates, diag=True, pred_noise=True)

        target = DataArray(
            mu,
            coords=[covariates.time],
            dims=['time'],
            attrs=self.dm.data.target.attrs,
        )
        alpha = (1 - ci)/2
        zscore = norm.ppf(1-alpha)

        cb = se**zscore

        # ax.scatter(samples.index, y, c='k', s=5, linewidths=0.5, edgecolors='white', zorder=3)

        target.plot.line(ax=ax, lw=1, zorder=2)

        ax.fill_between(
            target['time'], (target / cb), (target * cb), color='b', alpha=0.1, zorder=1
        )

        self.plot_observations(ax, zorder=3)

        # plt.scatter(samples.index, y, c='k', s=5, linewidths=0.5, edgecolors='white', zorder=3)
        # ax.fill_between(daily.index, (mu-ci), (mu+ci), color='b', alpha=.1, zorder=1 )

        # ax.plot(daily.index, mu, lw=1, zorder=2) #np.exp(mu)

        #

        return ax

    @is_fitted
    def plot_flux(self, covariates, ax: Axes = None):
        """Plot predicted flux versus time.

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
        mu, se = self.predict(covariates, diag=True, pred_noise=True)

        flux = self.compute_daily_flux(mu, covariates['flow'])
        flux_se = self.compute_daily_flux(se, covariates['flow'])

        ci = 1.96 * flux_se

        flux.plot.line(ax=ax, lw=1, zorder=2)

        ax.fill_between(
            flux['time'], (flux - ci), (flux + ci), color='b', alpha=0.1, zorder=1
        )

        return ax

    @is_fitted
    def observed_vs_predicted(self, ax: Axes = None):
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
    def contourf(self, ax: Axes = None):
        pass

    @is_fitted
    def countourf_diff(self, ax: Axes = None):
        pass

    def compute_daily_flux(self, concentration, flow):
        """Compute load from concentration and flow.

        Parameters
        ----------
        concentration : array_like
            Concentration.
        flow : array_like
            Flow.

        Returns
        -------
        array_like
            Load.
        """
        attrs = self.dm.data.target.attrs

        attrs.update({'units': 'kg/day'})
        SECONDS_TO_DAY = 86400
        MG_TO_KG = 1e-6
        L_TO_M3 = 1e-3

        a = concentration * MG_TO_KG * flow * SECONDS_TO_DAY * L_TO_M3
        da = DataArray(a, dims=['time'], coords=[flow.time], attrs=attrs)
        return da