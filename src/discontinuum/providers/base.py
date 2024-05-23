"""
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import Dataset


@dataclass
class MetaData:
    id: str
    name: str
    latitude: float
    longitude: float


def get_covariates(
        location: str, start_date: str, end_date: str, variable: str,
        ) -> Dataset:
    """Return timeseries of covariate data."""
    raise NotImplementedError


def get_target(
        location: str, start_date: str, end_date: str, variabe: str
        ) -> Dataset:
    """Return target data."""
    raise NotImplementedError


def get_metadata(location: str) -> MetaData:
    """Return metadata."""
    raise NotImplementedError
