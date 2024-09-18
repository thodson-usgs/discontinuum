import pytest
import json
import requests_mock
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.axes import Axes


from loadest_gp.providers import usgs
from loadest_gp import LoadestGPMarginalGPyTorch as LoadestGP
from loadest_gp.utils import concentration_to_flux, plot_annual_flux
from discontinuum.utils import time_substitution, aggregate_to_daily


site = "01491000" 
start_date = "1979-10-01"
end_date = "2011-09-30"

characteristic = 'Inorganic nitrogen (nitrate and nitrite)'
fraction = 'Dissolved'


@pytest.fixture
def demo_nwis_query_response():
    with open('tests/data/nwis_responses_loadest.json', 'r') as file:
        responses = json.load(file)
    return responses


def test_get_daily(demo_nwis_query_response):
    with requests_mock.Mocker() as m:
        m.get(demo_nwis_query_response['get_daily']['url'],
              text=demo_nwis_query_response['get_daily']['text'])
        m.get(demo_nwis_query_response['get_samples']['site_metadata']['url'],
              text=demo_nwis_query_response['get_samples']['site_metadata']['text'])
        data = usgs.get_daily(site=site,
                              start_date=start_date,
                              end_date=end_date)

    assert isinstance(data, xr.Dataset)
    assert hasattr(data, 'flow')
    assert hasattr(data, 'time')
    assert (data.time.size == data.flow.size)
    assert all(data.flow > 0)


def test_get_samples(demo_nwis_query_response):
    with requests_mock.Mocker() as m:
        m.get(demo_nwis_query_response['get_samples']['data']['url'],
              text=demo_nwis_query_response['get_samples']['data']['text'])
        m.get(demo_nwis_query_response['get_samples']['pcode']['url'],
              text=demo_nwis_query_response['get_samples']['pcode']['text'])
        m.get(demo_nwis_query_response['get_samples']['site_metadata']['url'],
              text=demo_nwis_query_response['get_samples']['site_metadata']['text'])
        data = usgs.get_samples(site=site,
                                start_date=start_date, 
                                end_date=end_date, 
                                characteristic=characteristic, 
                                fraction=fraction)

    assert isinstance(data, xr.Dataset)
    assert hasattr(data, 'concentration')
    assert hasattr(data, 'time')
    assert (data.time.size == data.concentration.size)
    assert all(data.concentration > 0)


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



@pytest.mark.filterwarnings("ignore:You have passed data")
def test_loadest_gp(demo_nwis_query_response):
    with requests_mock.Mocker() as m:
        m.get(demo_nwis_query_response['get_samples']['data']['url'],
              text=demo_nwis_query_response['get_samples']['data']['text'])
        m.get(demo_nwis_query_response['get_samples']['pcode']['url'],
              text=demo_nwis_query_response['get_samples']['pcode']['text'])
        m.get(demo_nwis_query_response['get_samples']['site_metadata']['url'],
              text=demo_nwis_query_response['get_samples']['site_metadata']['text'])
        samples = usgs.get_samples(site=site,
                                   start_date=start_date, 
                                   end_date=end_date, 
                                   characteristic=characteristic, 
                                   fraction=fraction)
        
        m.get(demo_nwis_query_response['get_daily']['url'],
              text=demo_nwis_query_response['get_daily']['text'])
        m.get(demo_nwis_query_response['get_samples']['site_metadata']['url'],
              text=demo_nwis_query_response['get_samples']['site_metadata']['text'])
        daily = usgs.get_daily(site=site,
                               start_date=start_date,
                               end_date=end_date)

    samples = aggregate_to_daily(samples)

    data = xr.merge([samples, daily], join='inner')

    model = LoadestGP()
    model.fit(target=data['concentration'],
              covariates=data[['time','flow']],
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
