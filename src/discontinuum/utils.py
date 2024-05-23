"""Data preprocessing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from xarray import Dataset


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


def time_substitution(ds: Dataset, interval: slice) -> Dataset:
    """Create a counterfactual by substituting data from an interval.

    Takes the data interval and repreats it to fill the whole dataset.
    For the dataset [1, 2, 3, 4, 5], `substitution(ds, slice(0, 0))`
    returns [1, 1, 1, 1, 1].

    Parameters
    ----------
    ds : Dataset
        Data to substitute.
    interval : slice
        Interval to substitute.

    Returns
    -------
    Dataset
        Data with substituted data.
    """
    out = ds.copy()
    ds_slice = ds.sel(time=interval)

    n_d = ds["time"].shape[0]
    n_s = ds_slice["time"].shape[0]

    rep = np.ceil(n_d / n_s)

    # for variable in dataset tile the slice and keep the first n_d values
    for var in ds.data_vars:
        out[var].values = np.tile(ds_slice[var].values, int(rep))[:n_d]

    return out
