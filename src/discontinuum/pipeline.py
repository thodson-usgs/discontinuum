from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xarray import DataArray


def zero_clip(a: ArrayLike) -> ArrayLike:
    """Clip an array to zero.

    Parameters
    ----------
    a : ArrayLike
        Array to clip.

    Returns
    -------
    ArrayLike
        Clipped array.
    """
    return np.clip(a, a_min=0, a_max=None)


def log_clip(a: ArrayLike) -> ArrayLike:
    """Clip an array to a small value.

    Parameters
    ----------
    a : ArrayLike
        Array to clip.

    Returns
    -------
    ArrayLike
        Clipped array.
    """
    return np.clip(a, a_min=1e-6, a_max=None)


def datetime_to_decimal_year(x: ArrayLike) -> ArrayLike:
    """Convert a timeseries to decimal year.

    Parameters
    ----------
    x : DataArray
        Timeseries to convert.

    Returns
    -------
    ArrayLike
        Decimal year array.
    """
    if not np.issubdtype(x.dtype, np.datetime64):
        raise ValueError("Array must contain numpy datetime64 objects.")

    # not flattening the array will cause an error
    dt = pd.to_datetime(x.flatten())

    julian = dt.to_julian_date()
    days_in_year = 365 + dt.is_leap_year
    start_of_year = pd.to_datetime(dt.year, format="%Y").to_julian_date()
    year = dt.year

    decimal_year = year + (julian - start_of_year)/(days_in_year)
    return decimal_year.to_numpy()


def decimal_year_to_datetime(x: ArrayLike) -> ArrayLike:
    """Convert a decimal year to a datetime.

    Parameters
    ----------
    x : ArrayLike
        Decimal year to convert.

    Returns
    -------
    ArrayLike
        Datetime array.
    """
    year = np.floor(x).astype(int)
    remainder = x - year

    start_of_year = pd.to_datetime(year, format="%Y")

    # Calculate the number of days in the year
    days_in_year = 365 + start_of_year.is_leap_year

    # Calculate the datetime corresponding to the decimal year
    dt = start_of_year + pd.to_timedelta(remainder * days_in_year, unit="D")

    return dt.round("1s").to_numpy()


class TimeTransformer:
    def __init__(self):
        """Transform time to decimal year."""
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return datetime_to_decimal_year(X)

    def inverse_transform(self, X):
        return decimal_year_to_datetime(X)


class MetadataManager(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self):
        """Store xarray metadata (attrs)."""
        self.attrs = None

    def fit(self, X, y=None):
        """Store metadata from a xarray DataArray.

        Parameters
        ----------
        X : DataArray

        y : None
            Ignored.
        """
        self.attrs = X.attrs
        self.name = X.name
        self.dims = X.dims

        return self

    def transform(self, X):
        """
        TODO should we use a separate tranformer for reshaping?

        Parameters
        ----------
        X : DataArray

        Returns
        -------
        Numpy array
        """
        return X.values.reshape(-1, 1)

    def inverse_transform(self, X):
        """Add xarray metadata to a numpy array.

        Parameters
        ----------
        X : Numpy array

        Returns
        -------
        DataArray
        """
        return DataArray(
            X.ravel(),
            attrs=self.attrs,
            name=self.name,
            dims=self.dims,
        )


class LogStandardPipeline(Pipeline):
    def __init__(self):
        """Pipeline for log-distributed data.
        """
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("clip", FunctionTransformer(func=log_clip)),
                ("log", FunctionTransformer(func=np.log, inverse_func=np.exp)),
                ("scaler", StandardScaler()),
            ]
        )


class StandardPipeline(Pipeline):
    def __init__(self):
        """Pipeline for normally distributed data.
        """
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("clip", FunctionTransformer(func=zero_clip,
                                             inverse_func=zero_clip)),
                ("scaler", StandardScaler()),
            ]
        )


class TimePipeline(Pipeline):
    def __init__(self):
        """Pipeline for time data."""
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("decimal_year", TimeTransformer()),
            ]
        )


class LogErrorPipeline(Pipeline):
    """Pipelin to transform error

    inverse_transform converts variance (in log space) to a GSE
    """

    def __init__(self):
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                # (
                #    "reshape",
                #    FunctionTransformer(
                #        func=np.reshape,
                #        inverse_func=np.reshape,
                #        kw_args={"newshape": (-1, 1)},
                #        inv_kw_args={"newshape": (1, -1)},
                #    ),
                # ),
                (
                    "log",
                    FunctionTransformer(
                        func=np.log, inverse_func=np.exp, check_inverse=False
                    ),
                ),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "square",
                    FunctionTransformer(
                        func=np.square, inverse_func=np.sqrt, check_inverse=False
                    ),
                ),
                (
                    "abs",
                    FunctionTransformer(
                        func=np.abs, inverse_func=np.abs, check_inverse=False
                    ),
                ),
            ]
        )
