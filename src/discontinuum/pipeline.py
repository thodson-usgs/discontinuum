from __future__ import annotations

import numpy as np
import pandas as pd


from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from scipy.stats import norm
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from xarray import DataArray


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
    # x = x.flatten()
    year = np.floor(x).astype(int)
    remainder = x - year

    start_of_year = pd.to_datetime(year, format="%Y")

    # Calculate the number of days in the year
    days_in_year = 365 + start_of_year.is_leap_year

    # Calculate the datetime corresponding to the decimal year
    dt = start_of_year + pd.to_timedelta(remainder * days_in_year, unit="D")

    return dt.round("1s").to_numpy()


class BaseTransformer(TransformerMixin, BaseEstimator):
    """Base class for transformers."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self


class ClipTransformer(BaseTransformer):
    """Clip a variable."""
    def __init__(self, min: float = None, max: float = None):
        """Clip a variable.

        Parameters
        ----------
        min : float
            Minimum value to clip.

        max : float
            Maximum value to clip.
        """
        self.min = min
        self.max = max

    def transform(self, X):
        return np.clip(X, a_min=self.min, a_max=self.max)

    def inverse_transform(self, X):
        return self.transform(X)


class LogTransformer(BaseTransformer):
    """Log-transform a variable."""
    def transform(self, X):
        return np.log(X)

    def inverse_transform(self, X):
        return np.exp(X)


class ShapeTransformer(OneToOneFeatureMixin, BaseTransformer):
    """Reshape a 1D array to 2D.

    Reshaping is a persistent issue that is still handled poorly.
    One issue is StandardScaler expects a 2D array, and a lot of
    the reshaping is to work around that.
    """
    def transform(self, X):
        return X.reshape(-1, 1)

    def inverse_transform(self, X):
        return X.squeeze()


class SquareTransformer(OneToOneFeatureMixin, BaseTransformer):
    """Square a variable."""
    def transform(self, X):
        return X**2

    def inverse_transform(self, X):
        return np.sqrt(X)


class StandardScaler(BaseTransformer):
    """Rescale a variable to have a mean of 0 and a standard deviation of 1.

    Reimplemens the sklearn.preprocessing.StandardScaler but removes the
    requirement of having 2D arrays.
    """
    def __init__(self, *, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        if self.with_mean:
            self.mean_ = X.mean(axis=0)
        if self.with_std:
            self.scale_ = X.std(axis=0)
        return self

    def transform(self, X):
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def inverse_transform(self, X):
        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X


class TimeTransformer(BaseTransformer):
    """Convert a datetime to decimal year."""
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
        return X.values

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
            X,
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
                ("reshape", ShapeTransformer()),
                ("clip", ClipTransformer(min=1e-6)),
                ("log", LogTransformer()),
                ("scaler", StandardScaler()),
            ]
        )


class NoOpPipeline(Pipeline):
    def __init__(self):
        """Pipeline that does not transform data.
        """
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("reshape", ShapeTransformer()),
                ("clip", ClipTransformer(min=0)),
            ]
        )


class StandardPipeline(Pipeline):
    def __init__(self):
        """Pipeline for normally distributed data.
        """
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("reshape", ShapeTransformer()),
                ("clip", ClipTransformer(min=0)),
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
                ("reshape", ShapeTransformer()),
                ("scaler", StandardScaler(with_std=False)),
            ]
        )


class ErrorPipeline(Pipeline, ABC):
    @abstractmethod
    def ci(self, mean, se, ci=0.95):
        """Calculate confidence interval for a variable.

        Parameters
        ----------
        mean : float
            Mean of the variable.
        se : float
            Standard error of the variable.
        ci : float
            Confidence level.

        Returns
        -------
        lower, upper : Tuple[float, float]
            Lower and upper bound of the confidence interval.
        """
        pass


class StandardErrorPipeline(ErrorPipeline):
    """Pipeline to transform error

    inverse_transform converts variance to SE.
    """

    def __init__(self):
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("reshape", ShapeTransformer()),
                ("scaler", StandardScaler(with_mean=False)),
                ("square", SquareTransformer()),
                # clip formerly used np.abs
                ("clip", ClipTransformer(min=0)),
            ]
        )

    def ci(self, mean, se, ci=0.95):
        """Calculate confidence interval for a standard variable.

        Parameters
        ----------
        mean : float
            Mean of the variable.
        se : float
            Standard error of the variable.
        ci : float
            Confidence level.

        Returns
        -------
        lower, upper : Tuple[float, float]
            Lower and upper bound of the confidence interval.
        """
        alpha = (1 - ci)/2
        zscore = norm.ppf(1-alpha)
        cb = se*zscore
        lower = mean - cb
        upper = mean + cb
        return lower, upper


class LogErrorPipeline(ErrorPipeline):
    """Pipelin to transform error

    inverse_transform converts variance (in log space) to a GSE
    """

    def __init__(self):
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("reshape", ShapeTransformer()),
                ("log", LogTransformer()),
                ("scaler", StandardScaler(with_mean=False)),
                ("square", SquareTransformer()),
                # clip formerly used np.abs
                ("clip", ClipTransformer(min=1e-6)),
            ]
        )

    def ci(self, mean, se, ci=0.95):
        """Calculate confidence interval for a log-transformed variable.

        Parameters
        ----------
        mean : float
            Mean of the variable.
        se : float
            Standard error of the variable.
        ci : float
            Confidence level.

        Returns
        -------
        lower, upper : Tuple[float, float]
            Lower and upper bound of the confidence interval.
        """
        alpha = (1 - ci)/2
        zscore = norm.ppf(1-alpha)
        cb = se**zscore
        lower = mean / cb
        upper = mean * cb
        return lower, upper
