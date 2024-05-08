"""Data preprocessing utilities."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import Dataset
    from typing import Dict

from functools import cached_property
from dataclasses import dataclass

import pandas as pd
import numpy as np

from discontinuum.transform import Transform, DesignMatrixTransform

# TODO create wrapper that validates input data
# TODO DataManager must track attrs


@dataclass
class DataManager:
    target: Dataset  # observations
    covariates: Dataset
    target_transform: Transform = Transform
    covariate_transforms: Dict[str, Transform] = None  # TODO optional

    def __post_init__(self):
        # setup transforms
        # self.covariates_transform = self.covariates_transform(self.covariates)
        self.covariate_transforms = DesignMatrixTransform(
            self.covariates, self.covariate_transforms
        )
        self.target_transform = self.target_transform(self.target)

    @cached_property
    def y(self):
        return np.array(self.target_transform.transform(self.target))

    @cached_property
    def X(self):
        return self.covariate_transforms.transform(self.covariates)

    def Xnew(self, ds: Dataset):
        return self.covariate_transforms.transform(ds)


# def decimal_year(t: Union[DatetimeIndex, DatetimeArray]) -> Series:
#    """Convert a pandas Datetime to decimal year.
#
#    Parameters
#    ----------
#    t : pandas.DatetimeIndex or pandas.DatetimeArray
#        Datetime to convert.
#
#    Returns
#    -------
#    pandas.Series
#        Decimal year.
#    """
#    days_in_year = 365 + t.is_leap_year
#    day_of_year = t.day_of_year
#    year = t.year
#    return year + day_of_year / days_in_year


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
