"""Data preprocessing utilities."""

from __future__ import annotations
from typing import TYPE_CHECKING

from functools import cached_property
from dataclasses import dataclass
import pandas as pd

if TYPE_CHECKING:
    from pandas import DatetimeIndex, DatetimeArray, Series, DataFrame
    from typing import Boolean, Dict, Union


@dataclass
class Parameter:
    name: str
    unit: str


@dataclass
class DataSet:
    data: DataFrame  # with time index
    metadata: Dict[str, Parameter]


@dataclass
class DataManager:
    target: DataSet  # observations
    features: DataSet

    def __post_init__(self):
        # setup transforms
        # transform features
        # transform target
        pass

    @cached_property
    def target_daily(self):
        return aggregate_to_daily(self.target.data)

    @cached_property
    def training_data(self):
        pass

    @cached_property
    def input_data(self):
        pass

    def new_data(self, data: DataSet = None):
        pass


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


def aggregate_to_daily(data: Union[DataFrame, Series], utc: Boolean = True) -> Series:
    """Aggregate data to daily values.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Data to aggregate.
    utc : bool, optional
        Whether to return the index as UTC. The default is True.

    Returns
    -------
    pandas.Series
        Daily aggregated data.
    """
    daily = data.groupby(data.index.date).mean()
    daily.index = pd.to_datetime(daily.index, utc=utc)
