from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xarray import DataArray
from discontinuum.pipeline import MetadataManager



class LogPropagation(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transformer for proper uncertainty propagation of log transforms
    
    Eq: f = ln(A) -> sigma_f = sigma_A / A
    """

    def fit(self, A):
        """Store data from A for later division in the transform.

        Parameters
        ----------
        A : ndarray
        """
        self.A_ = A

        return self

    def transform(self, X):
        """
        Propagate the uncertainty.

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        ndarray
        """
        return X / self.A_


class LogUncertaintyPipeline(Pipeline):
    """Pipeline to transform (propagate) uncertainty to log space"""

    def __init__(self):
        super().__init__(
            steps=[
                ("metadata", MetadataManager()),
                ("log", LogPropagation()),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "square",
                    FunctionTransformer(
                        func=np.square,
                        inverse_func=np.sqrt,
                        check_inverse=False,
                    ),
                ),
                (
                    "abs",
                    FunctionTransformer(
                        func=np.abs,
                        inverse_func=np.abs,
                        check_inverse=False,
                    ),
                ),
            ]
        )
