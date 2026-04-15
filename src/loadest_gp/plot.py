"""Plotting functions"""

from __future__ import annotations

from typing import TYPE_CHECKING

from discontinuum.engines.base import is_fitted
from discontinuum.plot import BasePlotMixin

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes
    from xarray import Dataset


class LoadestPlotMixin(BasePlotMixin):
    """Mixin plotting functions for Model class"""

    @is_fitted
    def plot_flux(self, covariates: Dataset, ax: Axes | None = None):
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

        ax.fill_between(flux["time"], (flux - ci), (flux + ci), color="b", alpha=0.1, zorder=1)

        return ax

    @is_fitted
    def contourf(
        self,
        covariate: str = "flow",
        ax: Axes | None = None,
        cbar_kwargs: dict | None = None,
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

        ax.set_yscale(y_scale)

        da = self.predict_grid(covariate=covariate, t_step=12)
        da.plot.contourf(x="time", y=covariate, ax=ax, cbar_kwargs=cbar_kwargs, **kwargs)

        return ax

    @is_fitted
    def countourf_diff(self, ax: Axes | None = None):
        pass
