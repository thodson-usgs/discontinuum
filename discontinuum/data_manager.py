"""Data preprocessing utilities."""

from __future__ import annotations
from typing import TYPE_CHECKING

from functools import cached_property
from dataclasses import dataclass
from xarray import DataArray, Dataset

import functools
import pandas as pd
import numpy as np


from discontinuum.pipeline import LogStandardPipeline, TimePipeline, LogErrorPipeline

from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from xarray import Dataset
    from typing import Dict
    from numpy.typing import ArrayLike


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

    target_pipeline: Pipeline = LogStandardPipeline
    error_pipeline: Pipeline = LogErrorPipeline
    covariate_pipelines: Dict[str, Pipeline] = None  # TODO optional

    def fit(self, target: Dataset, covariates: Dataset):
        """Initialize DataManager for a given data distribution."""
        # ensure time comes first in the dict

        default_pipeline = {'time': TimePipeline}
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