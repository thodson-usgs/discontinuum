import pytest
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.axes import Axes


from loadest_gp.providers import usgs
from loadest_gp import LoadestGPMarginalGPyTorch as LoadestGP
from loadest_gp.utils import concentration_to_flux, plot_annual_flux
from discontinuum.utils import time_substitution, aggregate_to_daily


@pytest.fixture
def sample_dataset():
    # Create a sample dataset for testing
    n = 20
    times = pd.date_range("2000-01-01", "2010-12-31", periods=n)
    flow = np.exp(np.random.randn(n))  # Random streamflow data
    concentration = np.exp(np.random.randn(n)) # Random solute concentration data

    ds = xr.Dataset(
        data_vars={
            'flow': ('time', flow),
            'concentration': ('time', concentration),
        },
        coords={'time': times},
    )

    return ds


@pytest.fixture
def daily_dataset():
    times = pd.date_range("2000-01-01", "2010-12-31", freq='d')
    flow = np.exp(np.random.rand(len(times)))  # Random streamflow data

    ds = xr.Dataset(
        data_vars={'flow': ('time', flow)},
        coords={'time': times},
      )
    
    return ds


def test_aggregate_to_daily():
    dates_hourly = pd.date_range('2000-01-01', '2000-02-01', freq='h')
    ds = xr.Dataset(data_vars={'data': ('time', np.ones(len(dates_hourly)))},
                    coords={'time': dates_hourly},)
    ds = aggregate_to_daily(ds)

    dates_daily = pd.date_range('2000-01-01', '2000-02-01', freq='d')

    assert all(ds.data == 1)
    assert all(ds.time == dates_daily)


def test_concentration_to_flux():
    dates = pd.date_range('2000-01-01', '2000-02-01', freq='d')
    ds = xr.Dataset(data_vars={'concentration': ('time', np.ones(len(dates))),
                               'flow': ('time', np.ones(len(dates)))},
                    coords={'time': dates})
    ds['concentration'].attrs['units'] = "mg/l"
    ds['concentration'].attrs['long_name'] = "Concentration"
    ds['flow'].attrs['units'] = "cubic meters per second"
    
    flux = concentration_to_flux(ds['concentration'], ds['flow'])

    assert isinstance(flux, xr.DataArray)
    assert hasattr(flux, 'time')
    assert all(flux.time == ds.time)
    
    assert isinstance(plot_annual_flux(flux), Axes)


def test_loadest_gp(sample_dataset):

    #data = sample_dataset()

    model = LoadestGP()
    model.fit(target=sample_dataset['concentration'],
              covariates=sample_dataset[['time','flow']],
              iterations=10)

    assert model.is_fitted
    assert isinstance(model.contourf(levels=5, y_scale='log'), Axes)  


def test_time_substitution():
    ds = xr.Dataset(data_vars={'data': ('time', np.linspace(0, 1, 10))},
                    coords={'time': np.arange(10)})
    slice_idx = np.random.randint(0, 10)
    ds_sub = time_substitution(ds, interval=slice(slice_idx, slice_idx))

    assert all(ds_sub.data == ds.data.isel(time=slice_idx).values)
    assert all(ds_sub.time == ds.time)
