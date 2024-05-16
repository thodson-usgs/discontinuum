"""Data preprocessing utilities."""

from __future__ import annotations
from typing import TYPE_CHECKING

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
