from __future__ import annotations
from typing import TYPE_CHECKING


from dataclasses import dataclass

if TYPE_CHECKING:
    # from pandas import DataFrame
    from xarray import DataArray, DataSet
    from typing import Optional, List, Union, Dict


@dataclass
class Parameter:
    standard_name: str
    units: str


@dataclass
class Location:
    id: str
    name: str
    latitude: float
    longitude: float


def get_daily(
    site: str, start_date: str, end_date: str, params: list[Parameter]
) -> DataSet:
    """Return timeseries of daily data for monitoring site."""
    raise NotImplementedError


def get_location(site: str) -> Location:
    """Return metadata for monitoring site."""
    raise NotImplementedError


def get_parameters() -> Parameter:
    """Return timeseries of paramete"""
    raise NotImplementedError


def get_samples(site: str, start_date: str, end_date: str, param: Parameter) -> DataSet:
    """Return sample data for monitoring site."""
    raise NotImplementedError
