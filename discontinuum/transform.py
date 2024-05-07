"""Data transformations to improve optimization"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xarray import Dataset

if TYPE_CHECKING:
    from typing import Dict
    from numpy.typing import ArrayLike
    from xarray import DataArray


class Transform:
    """Transformation class

    All children of Transform must have transfom and untransform methods
    """

    def __init__(self, x):
        """Create empty Transform object"""
        self.mean_ = None
        self.std_ = None
        _ = x

    def transform(self, x: ArrayLike) -> ArrayLike:
        """Transform data

        Parameters
        ----------
        x : array_like
            Data to be transformed.

        Returns
        -------
        ArrayLike
            Transformed data.
        """
        return x

    def untransform(self, z: ArrayLike) -> ArrayLike:
        return z


class ZTransform(Transform):
    """Z-transforms data to have zero mean and unit variance"""

    def __init__(self, x: ArrayLike):
        """Create a ZTransform object

        Parameters
        ----------
        x : array_like
          Data that defines the transform.
        """
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)

    def transform(self, x: ArrayLike) -> ArrayLike:
        """Transform to z score (standardize x)

        Parameters
        ----------
        x : array_like
          Data to be transformed.


        Returns
        -------
        ArrayLike
            original data standardized to zero mean and unit variance (z-score)
        """
        return (x - self.mean_) / self.std_

    def untransform(self, z: ArrayLike) -> ArrayLike:
        """Transform from z score back to original units.

        Parameters
        ----------
        z : array_like
          Transformed data

        Returns
        -------
        ArrayLike
            z-scores transformed back to original units.
        """
        return z * self.std_ + self.mean_


class LogZTransform(ZTransform):
    """Log transform then takes z-score."""

    def __init__(self, x: ArrayLike):
        """Create a LogZTransform for x.

        Parameters
        ----------
        x : array_like
          Data that defines the transform.
        """
        log_x = np.log(x)
        super().__init__(log_x)

    def transform(self, x: ArrayLike) -> ArrayLike:
        """Transform to log z-score

        Logs the data then standardizes to zero mean and unit variance.

        Parameters
        ----------
        x : array_like
          Data to be transformed.

        Returns
        -------
        ArrayLike
            log transform data standardized to zero mean and unit variance (log z-score)
        """
        log_x = np.log(x)
        return super().transform(log_x)

    def untransform(self, z: ArrayLike) -> ArrayLike:
        """Reverse log z-score transformation.

        Parameters
        ----------
        z : array_like
          Transformed data.

        Returns
        -------
        ArrayLike
            log z-scores transformed back to original units.
        """
        log_x = super().untransform(z)
        return np.exp(log_x)


class UnitTransform(Transform):
    """Transforms data to the unit (0 to 1) interval."""

    def __init__(self, x: ArrayLike):
        """Create UnitTransform of array

        Parameters
        ----------
        x : array_like
            Data that defines the transform.
        """
        self.max_ = np.nanmax(x, axis=0)

    def transform(self, x: ArrayLike) -> ArrayLike:
        """Transform to unit interval

        Parameters
        ----------
        x : array_like
            Data to be transformed.

        Returns
        -------
        ArrayLike
            Original data transformed to unit interval
        """
        return x / self.max_

    def untransform(self, z: ArrayLike) -> ArrayLike:
        """Transform from unit interval back to original units

        Parameters
        ----------
        z : array_like
          Transformed data.

        Returns
        -------
        ArrayLike
            Unit interval transformed back to original units.
        """
        return z * self.max_


class DecimalYearTransform(Transform):
    """TODO FIX untransform returns the wrong date"""

    def __init__(self, x: DataArray = None):
        pass

    def transform(self, x: DataArray) -> ArrayLike:
        """Convert a timeseries to decimal year.

        Parameters
        ----------
        x : DataArray
            Timeseries to convert.
        """
        days_in_year = 365 + x.dt.is_leap_year
        day_of_year = x.dt.dayofyear
        year = x.dt.year
        decimal_year = year + day_of_year / days_in_year
        return decimal_year.to_numpy()

    def untransform(self, z: ArrayLike) -> ArrayLike:
        """TODO FIX"""
        year = np.floor(z)
        dt = pd.to_datetime(year, format="%Y")
        days_in_year = 365 + dt.is_leap_year
        day_of_year = np.floor((z - year) * days_in_year) + 1
        dt = dt + pd.to_timedelta(day_of_year, unit="D")
        return pd.to_datetime(dt.date)  # remove decimal part


class DesignMatrixTransform(Transform):
    """Transforms data to a design matrix"""

    def __init__(self, x: Dataset, index: str, transforms: Dict[str, Transform]):
        self._transforms = {}
        self._index = index

        for key, transform in transforms.items():
            self._transforms[key] = transform(x[key])

    def transform(self, x: Dataset) -> ArrayLike:
        """Transform data to design matrix

        Parameters
        ----------
        x : Dataset
            Data to be transformed.

        Returns
        -------
        ArrayLike
            Design matrix.
        """
        index_len = x.sizes.get(self._index)
        design_matrix = np.empty((index_len, len(self._transforms)))

        for i, (key, transform) in enumerate(self._transforms.items()):
            design_matrix[:, i] = transform.transform(x[key])

        return design_matrix

    def untransform(self, z: ArrayLike) -> Dataset:
        """Transform design matrix back to original data

        Parameters
        ----------
        z : ArrayLike
            Design matrix.

        Returns
        -------
        Dataset
            Original data.
        """
        data = {}
        for i, (key, transform) in enumerate(self._transforms.items()):
            data[key] = transform.untransform(z[:, i])

        return Dataset(data)
