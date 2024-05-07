"""Data preprocessing utilities."""

from __future__ import annotations
from typing import TYPE_CHECKING

from functools import cached_property
from dataclasses import dataclass

import pandas as pd

if TYPE_CHECKING:
    from xarray import Dataset
    from pandas import DatetimeIndex, DatetimeArray, Series, DataFrame
    from typing import Boolean, Dict, Union
    from discontinuum.transform import Transform


@dataclass
class Parameter:
    name: str
    unit: str


@dataclass
class DataManager:
    target: Dataset  # observations
    features: Dataset
    target_transform: Transform = None
    feature_transform: Transform = None

    def __post_init__(self):
        # setup transforms
        # transform features
        # transform target
        pass

    @cached_property
    def y(self):
        self.target_transform.transform(self.target)

    @cached_property
    def X(self):
        self.feature_transform.transform(self.features)

    def Xnew(self, ds: Dataset):
        return self.feature_transform.transform(ds)


def decimal_year(t: Union[DatetimeIndex, DatetimeArray]) -> Series:
    """Convert a pandas Datetime to decimal year.

    Parameters
    ----------
    t : pandas.DatetimeIndex or pandas.DatetimeArray
        Datetime to convert.

    Returns
    -------
    pandas.Series
        Decimal year.
    """
    days_in_year = 365 + t.is_leap_year
    day_of_year = t.day_of_year
    year = t.year
    return year + day_of_year / days_in_year


def aggregate_to_daily(ds: Dataset) -> Dataset:
    """Aggregate data to daily values.

    Parameters
    ----------
    ds : Dataset
        Data to aggregate.

    Returns
    -------
    Dataset
        Daily aggregated data.
    """
    daily = ds.groupby(ds.time.dt.date).mean()
    # the groupy changes the coordinate name and type,
    # so we convert it back
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename({"date": "time"})

    return daily
