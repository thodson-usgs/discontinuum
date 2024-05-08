"""Plotting functions"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from discontinuum.models import is_fitted

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
    def concentration(self, ax: Axes = None):
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

        return ax

    @is_fitted
    def flux(self, ax: Axes = None):
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
