from __future__ import annotations
from typing import TYPE_CHECKING


from dataclasses import dataclass

if TYPE_CHECKING:
    # from pandas import DataFrame
    from xarray import DataArray, DataSet
    from typing import Optional, List, Union, Dict


@dataclass
class Covariate:
    standard_name: str
    units: str

@dataclass
class Location:
    id: str
    name: str
    latitude: float
    longitude: float


def get_timeseries(
    site: str, start_date: str, end_date: str, params: list[Covariate]
) -> DataSet:
    """Return timeseries of daily data for monitoring site."""
    raise NotImplementedError


def get_location(site: str) -> Location:
    """Return metadata for monitoring site."""
    raise NotImplementedError


def get_covariates() -> Covariates:
    """Return timeseries of paramete"""
    raise NotImplementedError


def get_target(location: str, start_date: str, end_date: str, target: Covariate) -> DataSet:
    """Return sample data for monitoring site."""
    raise NotImplementedError
