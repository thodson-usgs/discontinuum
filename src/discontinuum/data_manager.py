"""Data preprocessing utilities."""

from __future__ import annotations

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
    from numpy.typing import ArrayLike


@dataclass
class Data:
    target: Dataset
    covariates: Dataset
    target_unc: Dataset = None


@dataclass
class DataManager:
    """Manages data transformations between raw and model space."""

    target_pipeline: type[Pipeline] = LogStandardPipeline
    error_pipeline: type[Pipeline] = LogErrorPipeline
    covariate_pipelines: dict[str, Pipeline] = None

    def fit(self, target: Dataset, covariates: Dataset, target_unc: Dataset = None):
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

        # Invalidate cached properties so they are recomputed
        for attr in ("y", "y_unc", "X"):
            self.__dict__.pop(attr, None)

        # Only fit pipelines if they are classes (not already fitted instances)
        if isinstance(self.target_pipeline, type):
            self.target_pipeline = self.target_pipeline().fit(target)
        if isinstance(self.error_pipeline, type):
            self.error_pipeline = self.error_pipeline().fit(target)

        for key, value in self.covariate_pipelines.items():
            if isinstance(value, type):
                self.covariate_pipelines[key] = value().fit(covariates[key])

    def transform_covariates(self, covariates: Dataset) -> ArrayLike:
        """Transform covariates into design matrix."""
        coords_shape = tuple(s for coord in covariates.coords for s in covariates.coords[coord].shape)
        X = np.empty(coords_shape + (len(self.covariate_pipelines),))
        for i, (key, value) in enumerate(self.covariate_pipelines.items()):
            X[..., i] = value.transform(covariates[key]).flatten()
        return X

    def inverse_transform_covariates(self, X: ArrayLike) -> Dataset:
        """Inverse transform design matrix into covariates."""
        covariates = {}
        for i, (key, value) in enumerate(self.covariate_pipelines.items()):
            covariates[key] = value.inverse_transform(X[:, i])
        return Dataset(covariates)

    @cached_property
    def y(self) -> ArrayLike:
        """Transform target to model space."""
        return self.target_pipeline.transform(self.data.target).flatten()

    @cached_property
    def y_unc(self) -> ArrayLike:
        """Transform target uncertainty to model space."""
        return self.error_pipeline.transform(self.data.target_unc).flatten()

    @cached_property
    def X(self) -> ArrayLike:
        """Transform covariates to model space."""
        return self.transform_covariates(self.data.covariates)

    def Xnew(self, ds: Dataset) -> ArrayLike:
        """Transform new covariates to model space."""
        return self.transform_covariates(ds)

    def y_t(self, y: ArrayLike) -> Dataset:
        """Inverse transform target from model space."""
        return self.target_pipeline.inverse_transform(y)

    def get_dim(self, dim: str) -> int:
        """Get the column index of a variable in the design matrix.

        Parameters
        ----------
        dim : str
            Dimension name.

        Returns
        -------
        int
            Column index in design matrix.
        """
        cov_list = list(self.data.covariates.coords) + list(self.data.covariates)

        return cov_list.index(dim)
