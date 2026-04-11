from __future__ import annotations

from typing import TYPE_CHECKING

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass

from discontinuum.data_manager import DataManager
from discontinuum.pipeline import (
    LogErrorPipeline,
    LogStandardPipeline,
    StandardErrorPipeline,
    StandardPipeline,
)

if TYPE_CHECKING:
    from typing import Dict, Literal, Optional
    from xarray import DataArray, Dataset


@dataclass
class ModelConfig:
    """Configuration for model data transformations."""
    transform: Literal["log", "standard"] = "log"


class BaseModel(ABC):
    def __init__(self, model_config: Optional[Dict] = None):
        """Base class for all models.

        Parameters
        ----------
        model_config : dict, optional
            Configuration for the model. The default is None.
        """
        if model_config is None:
            model_config = {}

        self.model_config = model_config
        self.dm = None
        self.is_fitted = False

    @abstractmethod
    def fit(self,
            covariates: Dataset,
            target: Dataset,
            **kwargs,
            ):
        """Fit model to data.

        Parameters
        ----------
        covariates : Dataset
            Covariates for training.
        target : Dataset
            Target data for training.
        kwargs : dict
            Additional keyword arguments.
        """
        self.is_fitted = True
        return self

    @abstractmethod
    def predict(self, covariates: Dataset) -> DataArray:
        """Use a fitted model to make predictions on new data.

        Parameters
        ----------
        covariates : Dataset
            Covariates for prediction.
        """
        pass

    @abstractmethod
    def build_model(self, X, y):
        pass

    @abstractmethod
    def build_datamanager(self):
        """Build DataManager for the model."""
        pass


class DataMixin:
    """Shared logic for building a DataManager with log/standard transforms."""

    def _build_datamanager(
            self,
            covariate_pipelines: Dict,
            model_config: ModelConfig = ModelConfig(),
            ):
        if model_config.transform == "log":
            target_pipeline = LogStandardPipeline
            error_pipeline = LogErrorPipeline
        elif model_config.transform == "standard":
            target_pipeline = StandardPipeline
            error_pipeline = StandardErrorPipeline
        else:
            raise ValueError(
                "Model config transform must be 'log' or 'standard'."
            )

        self.dm = DataManager(
            target_pipeline=target_pipeline,
            error_pipeline=error_pipeline,
            covariate_pipelines=covariate_pipelines,
        )


def is_fitted(func):
    """Decorator checks whether model has been fit."""

    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if not self.is_fitted:
            raise RuntimeError(
                "The model hasn't been fitted yet, call .fit()."
            )
        return func(self, *args, **kwargs)

    return inner
