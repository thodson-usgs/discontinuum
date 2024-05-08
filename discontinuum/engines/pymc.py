"""Data transformations to improve optimization"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict

import pymc as pm

from discontinuum.models.base import BaseModel, is_fitted


class PyMCModel(BaseModel):
    def __init__(
        self,
        model_config: Dict = None,
    ):
        """ """
        self.model_config = model_config
        self.is_fitted = False

    def fit(self, X, y=None):
        """Fit the model to data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target values.
        """
        self.is_fitted = True
        # preprocess data; data manager
        self.X = X
        self.y = y

        self.build_model(self.X, self.y)

        with self.model:
            self.mp = pm.find_MAP(method="BFGS")

    @is_fitted
    def predict(self, Xnew, diag=True, pred_noise=True):
        """Uses the fitted model to make predictions on new data."""
        with self.model:
            mu, cov = self.gp.predict(
                Xnew, point=self.mp, diag=diag, pred_noise=pred_noise
            )
        # return self.gp.predict(Xnew, point=self.mp, diag=diag, pred_noise=pred_noise)
        return mu, cov

    def build_model(self, X, y, **kwargs):
        """
        Creates an instance of pm.Model based on provided data and model_config, and
        attaches it to self.

        The subclass method must instantiate self.model and self.gp.

        Raises
        ------
        NotImplementedError
        """
        self.model = None
        self.gp = None

        raise NotImplementedError("This method must be implemented in a subclass")
