"""Data transformations to improve optimization"""

from __future__ import annotations
from math import e
from typing import TYPE_CHECKING

import pymc as pm
import numpy as np

from xarray import DataArray


from discontinuum.engines.base import BaseModel, is_fitted


if TYPE_CHECKING:
    from typing import Dict


class MarginalGP(BaseModel):
    def __init__(
        self,
        model_config: Dict = {},
    ):
        """ """
        self.dm = None  # DataManager
        self.model_config = model_config
        self.is_fitted = False

    def fit(self, covariates, target=None):
        """Fit the model to data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target values.
        """
        self.is_fitted = True
        # preprocessing: setup data manager
        self.dm.fit(target=target, covariates=covariates)
        self.X = self.dm.X
        self.y = self.dm.y

        self.model = self.build_model(self.X, self.y)

        self.mp = pm.find_MAP(method="BFGS", model=self.model)

    @is_fitted
    def predict(self, covariates, diag=True, pred_noise=False) -> DataArray:
        """Uses the fitted model to make predictions on new data."""
        Xnew = self.dm.Xnew(covariates)

        mu, cov = self.gp.predict(
            Xnew, point=self.mp, diag=diag, pred_noise=pred_noise, model=self.model
        )

        target = self.dm.y_t(mu)
        # TODO the reshape should be done in the pipeline
        se = self.dm.error_pipeline.inverse_transform(cov.reshape(-1, 1))

        return target, se
    
    def sample(self, covariates, n=1000, diag=False, pred_noise=False, method='cholesky', tol=1e-6) -> DataArray:
        """Sample from the posterior distribution of the model.

        Parameters
        ----------
        covariates : Dataset
            Covariates for prediction.
        n : int, optional
            Number of samples to draw.
        method{ ‘svd’, ‘eigh’, ‘cholesky’}, optional
            Method to use for covariance matrix decomposition. The default is ‘cholesky’.
        tol : float, optional
            Tolerance when checking the singular values in covariance matrix. The default is 1e-6.
        """
        Xnew = self.dm.Xnew(covariates)
        mu, cov = self.gp.predict(
            Xnew, point=self.mp, diag=diag, pred_noise=pred_noise, model=self.model
        )

        rng = np.random.default_rng()
        sim = rng.multivariate_normal(mu, cov, size=n, method=method, tol=tol)

        # TODO modify transform to handle samples/draws HACK
        temp = self.dm.y_t(sim)
        data = temp.data.reshape(n, -1)
        attrs = temp.attrs
        da = DataArray(
            data,
            coords={"time" : covariates.time, "draw" : np.arange(n)},
            dims=["draw", "time"],
            attrs=attrs,
        )

        return da
    

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
