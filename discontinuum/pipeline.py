from __future__ import annotations
from typing import TYPE_CHECKING


import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

from xarray import DataArray


from numpy.typing import ArrayLike


zero_clip = lambda a: np.clip(a, a_min=0, a_max=None)

log_clip = lambda a: np.clip(a, a_min=1e-6, a_max=None)


def time_to_decimal_year(x: ArrayLike) -> ArrayLike:
    """Convert a timeseries to decimal year.

    Parameters
    ----------
    x : DataArray
        Timeseries to convert.
    """
    # not flattening the array will cause an error
    dt = pd.to_datetime(x.flatten())

    days_in_year = 365 + dt.is_leap_year
    day_of_year = dt.dayofyear
    year = dt.year
    decimal_year = year + day_of_year / days_in_year
    return decimal_year.to_numpy()


def decimal_year_to_time(z: ArrayLike) -> ArrayLike:
    """TODO FIX"""
    z = z.flatten()
    year = np.floor(z)
    dt = pd.to_datetime(year, format="%Y")
    days_in_year = 365 + dt.is_leap_year
    day_of_year = np.floor((z - year) * days_in_year) + 1
    dt = dt + pd.to_timedelta(day_of_year, unit="D")
    return pd.to_datetime(dt.date)  # remove decimal part


# class TimeTransformer(TransformerMixin, BaseEstimator):
class TimeTransformer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return time_to_decimal_year(X)

    def inverse_transform(self, X):
        return decimal_year_to_time(X)


class MetadataManager(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self):
        self.attrs = None

    def fit(self, X, y=None):
        """

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

        Parameters
        ----------
        X : DataArray

        Returns
        -------
        Numpy array
        """
        return X.values.reshape(-1, 1)

    def inverse_transform(self, X):
        """

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
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("clip", FunctionTransformer(func=zero_clip, inverse_func=zero_clip)),
                ("scaler", StandardScaler()),
            ]
        )


class TimePipeline(Pipeline):
    def __init__(self):
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
