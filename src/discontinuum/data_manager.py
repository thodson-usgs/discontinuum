"""Data preprocessing utilities."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from sklearn.pipeline import Pipeline
from xarray import Dataset

from discontinuum.pipeline import (
    LogErrorPipeline,
    LogStandardPipeline,
)

if TYPE_CHECKING:
    from typing import Dict, Type

    from numpy.typing import ArrayLike


def is_initialized(func):
    """Decorator checks whether model has been fit."""

    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if not self.is_initialized:
            raise RuntimeError(
                "The DataManager has not been initialized,",
                "call .init(target, covariates)."
            )
        return func(self, *args, **kwargs)

    return inner


@dataclass
class Data:
    target: Dataset
    covariates: Dataset
    target_unc: Dataset = None


# TODO create wrapper that validates input data
# TODO DataManager must track attrs
@dataclass
class DataManager:
    """ """

    target_pipeline: Type[Pipeline] = LogStandardPipeline
    error_pipeline: Type[Pipeline] = LogErrorPipeline
    covariate_pipelines: Dict[str, Pipeline] = None

    def fit(
            self,
            target: Dataset,
            covariates: Dataset,
            target_unc: Dataset = None):
        """Initialize DataManager for a given data distribution.

        Parameters
        ----------
        target : Dataset
            Target data.

        covariates : Dataset
            Covariate data.

        target_unc : Dataset
            Target uncertainty. Default is None.
        """
        self.data = Data(target, covariates, target_unc)

        self.target_pipeline = self.target_pipeline().fit(target)
        self.error_pipeline = self.error_pipeline().fit(target)

        for key, value in self.covariate_pipelines.items():
            self.covariate_pipelines[key] = value().fit(covariates[key])

    def transform_covariates(self, covariates: Dataset) -> ArrayLike:
        """Transform covariates into design matrix"""
        coords_shape = tuple()
        for coord in covariates.coords:
            coords_shape += covariates.coords[coord].shape
        X = np.empty(coords_shape + (len(self.covariate_pipelines), ))
        for i, (key, value) in enumerate(self.covariate_pipelines.items()):
            X[..., i] = value.transform(covariates[key]).flatten()
        return X

    # TODO handle reshaping in pipeline
    def inverse_transform_covariates(self, X: ArrayLike) -> Dataset:
        """Inverse transform design matrix into covariates"""
        # inverse transform each column of X using the corresponding pipeline
        covariates = {}
        for i, (key, value) in enumerate(self.covariate_pipelines.items()):
            covariates[key] = value.inverse_transform(X[:, i])
        return Dataset(covariates)

    @cached_property
    def y(self, dtype="float32") -> ArrayLike:
        """Convenience function for DataManager.target.transform"""
        return self.target_pipeline.transform(self.data.target).flatten()

    @cached_property
    def y_unc(self, dtype="float32") -> ArrayLike:
        """Convenience function for DataManager.target.transform"""
        return self.error_pipeline.transform(self.data.target_unc).flatten()

    @cached_property
    def X(self) -> ArrayLike:
        """Convenience function for DataManager.covariates.transform"""
        return self.transform_covariates(self.data.covariates)

    def Xnew(self, ds: Dataset) -> ArrayLike:
        """Convenience function for DataManager.covariates.transform"""
        return self.transform_covariates(ds)

    def y_t(self, y: ArrayLike) -> Dataset:
        """Convenience function for DataManager.target.untransform"""
        return self.target_pipeline.inverse_transform(y)

    def get_dim(self, dim: str) -> int:
        """Get the dimension of a variable.

        In other words, its column in the design matrix.

        Parameters
        ----------
        dim : str
            Dimension name.

        Returns
        -------
        int
            Dimension (column) in design matrix.
        """
        # Coords come first in Pipelines
        cov_list = (list(self.data.covariates.coords)
                    + list(self.data.covariates))

        return cov_list.index(dim)
