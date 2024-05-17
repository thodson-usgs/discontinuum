from __future__ import annotations
from typing import TYPE_CHECKING


from dataclasses import dataclass

if TYPE_CHECKING:
    # from pandas import DataFrame
    from xarray import Dataset
    from typing import Optional, List, Union, Dict


@dataclass
class MetaData:
    id: str
    name: str
    latitude: float
    longitude: float


def get_timeseries(
    location: str, start_date: str, end_date: str, variable: str
) -> Dataset:
    """Return timeseries of daily data for monitoring site."""
    raise NotImplementedError

def get_target(location: str, start_date: str, end_date: str, variabe: str) -> Dataset:
    """Return target data."""
    raise NotImplementedError

def get_metadata(location: str) -> MetaData:
    """Return metadata."""
    raise NotImplementedError
