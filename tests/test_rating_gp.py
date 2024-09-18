import pytest
import json
import requests_mock
from xarray import Dataset
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

from rating_gp.providers import usgs
from rating_gp.models.gpytorch import RatingGPMarginalGPyTorch as RatingGP


site = '10154200'
start_date = "1988-10-01" 
end_date = "2021-09-30"


@pytest.fixture
def demo_nwis_query_response():
    with open('tests/data/nwis_responses_rating.json', 'r') as file:
        responses = json.load(file)
    return responses


def test_get_measurements(demo_nwis_query_response):
    with requests_mock.Mocker() as m:
        m.get(requests_mock.ANY,
              text=demo_nwis_query_response['get_measurements']['text'])
        data = usgs.get_measurements(site=site,
                                     start_date=start_date,
                                     end_date=end_date)

    assert isinstance(data, Dataset)
    assert hasattr(data, 'stage')
    assert hasattr(data, 'discharge')
    assert hasattr(data, 'discharge_unc')
    assert hasattr(data, 'time')
    assert ((data.time.size == data.stage.size)
            & (data.time.size == data.discharge.size)
            & (data.time.size == data.discharge_unc.size))
    assert all(data.stage > 0)
    assert all(data.discharge > 0)
    # Geometric error value is always >= 1
    assert all(data.discharge_unc >= 1)


@pytest.mark.filterwarnings("ignore:The input matches the stored training data")
@pytest.mark.filterwarnings("ignore:You have passed data")
# @pytest.mark.parametrize("site", ['12413470', '10131000', '09261000', '10154200'])
def test_rating_gp(demo_nwis_query_response):
    with requests_mock.Mocker() as m:
        m.get(requests_mock.ANY,
              text=demo_nwis_query_response['get_measurements']['text'])
        data = usgs.get_measurements(site=site,
                                     start_date=start_date,
                                     end_date=end_date)

    model = RatingGP()
    model.fit(target=data['discharge'],
              covariates=data[['stage']],
              target_unc=data['discharge_unc'],
              iterations=10)
    assert model.is_fitted

    assert isinstance(model.plot_stage(), Axes)
    assert isinstance(model.plot_discharge(), Axes)
    assert isinstance(model.plot_observed_rating(), Axes)
    assert isinstance(model.plot_rating(covariates=data[['stage']]), Axes)
    ax = model.plot_ratings_in_time(
        time=pd.date_range('1990', '2021', freq='5YS-OCT'), ci=0.95
    )
    assert isinstance(ax, Axes)
    assert isinstance(model.add_time_colorbar(ax=ax), Colorbar)
