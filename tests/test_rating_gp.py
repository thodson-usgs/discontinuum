import pytest
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

from rating_gp.providers import usgs
from rating_gp.models.gpytorch import RatingGPMarginalGPyTorch as RatingGP


@pytest.fixture
def training_data():
    # Create a sample dataset for testing
    n = 20
    times = pd.date_range("2000-01-01", "2010-12-31", periods=n)
    discharge = np.exp(np.random.randn(n))  # Random streamflow data
    discharge_unc = 1 + 0.1 * np.random.rand(n)  # Random uncertainty data
    stage = np.random.randn(n) # Random stage data

    ds = xr.Dataset(
        data_vars={
            'discharge': ('time', discharge),
            'discharge_unc': ('time', discharge_unc),
            'stage': ('time', stage),
        },
        coords={'time': times},
    )

    return ds


def test_rating_gp(training_data):

    model = RatingGP()
    model.fit(target=training_data['discharge'],
              covariates=training_data[['stage']],
              target_unc=training_data['discharge_unc'],
              iterations=10)
    assert model.is_fitted

    assert isinstance(model.plot_stage(), Axes)
    assert isinstance(model.plot_discharge(), Axes)
    assert isinstance(model.plot_observed_rating(), Axes)
    assert isinstance(model.plot_rating(covariates=training_data[['stage']]), Axes)
    ax = model.plot_ratings_in_time(
        time=pd.date_range('1990', '2021', freq='5YS-OCT'), ci=0.95
    )
    assert isinstance(ax, Axes)
    assert isinstance(model.add_time_colorbar(ax=ax), Colorbar)
