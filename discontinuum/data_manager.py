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
    TimePipeline,
)

if TYPE_CHECKING:
    from typing import Dict, Type

    from numpy.typing import ArrayLike
    from xarray import Dataset


def is_initialized(func):
    """Decorator checks whether model has been fit."""

    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if not self.is_initialized:
            raise RuntimeError(
                "The DataManager has not been initialized, call .init(target, covariates)."
            )
        return func(self, *args, **kwargs)

    return inner


@dataclass
class Data:
    target: Dataset
    covariates: Dataset


# TODO create wrapper that validates input data
# TODO DataManager must track attrs
@dataclass
class DataManager:
    """ """

    target_pipeline: Type[Pipeline] = LogStandardPipeline
    error_pipeline: Type[Pipeline] = LogErrorPipeline
    covariate_pipelines: Dict[str, Pipeline] = None

    def fit(self, target: Dataset, covariates: Dataset):
        """Initialize DataManager for a given data distribution."""
        # ensure time comes first in the dict

        default_pipeline = {"time": TimePipeline}
        default_pipeline.update(self.covariate_pipelines)
        self.covariate_pipelines = default_pipeline

        self.data = Data(target, covariates)

        self.target_pipeline = self.target_pipeline().fit(target)
        self.error_pipeline = self.error_pipeline().fit(target)

        for key, value in self.covariate_pipelines.items():
            self.covariate_pipelines[key] = value().fit(covariates[key])

    def transform_covariates(self, covariates: Dataset) -> ArrayLike:
        """Transform covariates into design matrix"""
        X = np.empty((len(covariates.time), len(self.covariate_pipelines)))
        for i, (key, value) in enumerate(self.covariate_pipelines.items()):
            X[:, i] = value.transform(covariates[key]).flatten()
        return X

    # TODO handle reshaping in pipeline
    def inverse_transform_covariates(self, X: ArrayLike) -> Dataset:
        """Inverse transform design matrix into covariates"""
        # inverse transform each column of X using the corresponding pipeline
        covariates = {}
        for i, (key, value) in enumerate(self.covariate_pipelines.items()):
            covariates[key] = value.inverse_transform(X[:, i].reshape(1, -1))
        return Dataset(covariates)

    @cached_property
    def y(self) -> ArrayLike:
        """Convenience function for DataManager.target.transform"""
        return self.target_pipeline.transform(self.data.target).flatten()

    @cached_property
    def X(self) -> ArrayLike:
        """Convenience function for DataManager.covariates.transform"""
        return self.transform_covariates(self.data.covariates)

    def Xnew(self, ds: Dataset) -> ArrayLike:
        """Convenience function for DataManager.covariates.transform"""
        return self.transform_covariates(ds)

    def y_t(self, y: ArrayLike) -> Dataset:
        """Convenience function for DataManager.target.untransform"""
        return self.target_pipeline.inverse_transform(y.reshape(-1, 1))

    def get_dim(self, dim: str, index="time") -> int:
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
        cov_list = [index] + list(self.data.covariates)

        return cov_list.index(dim)
