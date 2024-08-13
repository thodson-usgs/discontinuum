from __future__ import annotations

from typing import TYPE_CHECKING

import functools
from abc import ABC, abstractmethod

from discontinuum.data_manager import DataManager
from discontinuum.pipeline import StandardErrorPipeline, StandardPipeline

if TYPE_CHECKING:
    from typing import Dict, Optional
    from xarray import DataArray, Dataset


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
        """
        TODO Sometimes this sets self.model and sometimes this returns model.
        Not sure that we can standardize the behavior for different engines.
        """
        pass

    @abstractmethod
    def build_datamanager(self):
        """Build DataManager for the model."""
        self.dm = DataManager(
            target_pipeline=StandardPipeline,
            error_pipeline=StandardErrorPipeline,
            covariate_pipelines=StandardPipeline,
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
