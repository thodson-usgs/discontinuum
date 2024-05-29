"""Data transformations to improve optimization"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
from xarray import DataArray

from discontinuum.engines.base import BaseModel, is_fitted

if TYPE_CHECKING:
    from typing import Dict, Optional

    from xarray import Dataset


class LatentGP(BaseModel):
    def __init__(
        self,
        model_config: Optional[Dict] = None,
    ):
        """ """
        super().__init__(model_config=model_config)

    def fit(self, covariates, target=None):
        pass


class MarginalGP(BaseModel):
    def __init__(
        self,
        model_config: Optional[Dict] = None,
    ):
        """ """
        super().__init__(model_config=model_config)

    def fit(self, covariates: Dataset, target: Dataset, method: str = "BFGS"):
        """Fit the model to data.

        Parameters
        ----------
        covariates : Dataset
            Covariates for prediction.
        target : Dataset
            Target data for prediction.
        method : str, optional
            Optimization method. The default is "BFGS".
        """
        self.is_fitted = True
        # preprocessing: setup data manager
        self.dm.fit(target=target, covariates=covariates)
        self.X = self.dm.X
        self.y = self.dm.y

        self.model = self.build_model(self.X, self.y)

        self.mp = pm.find_MAP(method=method, model=self.model)

    @is_fitted
    def predict(self, covariates, diag=True, pred_noise=False) -> DataArray:
        """Uses the fitted model to make predictions on new data."""
        Xnew = self.dm.Xnew(covariates)

        mu, cov = self.gp.predict(
            Xnew,
            point=self.mp,
            diag=diag,
            pred_noise=pred_noise,
            model=self.model,
        )

        target = self.dm.y_t(mu)
        # TODO the reshape should be done in the pipeline
        se = self.dm.error_pipeline.inverse_transform(cov.reshape(-1, 1))

        return target, se

    @is_fitted
    def predict_grid(self, covariate: str, index="time", t_step=12):
        """Predict on a grid of points.

        Parameters
        ----------
        covariate_dim : int, optional
            Dimension to predict on. The default is 1.
        t_step : int, optional
            Time steps per year. The default is 12.
        """
        time_dim = self.dm.get_dim(index)
        covariate_dim = self.dm.get_dim(covariate)

        x_max = self.dm.X.max(axis=0)
        x_min = self.dm.X.min(axis=0)
        x_range = x_max - x_min

        n_cov = 18
        n_time = np.round(x_range[time_dim] * t_step).astype(int)

        x_time = np.linspace(x_min[time_dim], x_max[time_dim], n_time)
        x_cov = np.linspace(x_min[covariate_dim], x_max[covariate_dim], n_cov)

        # TODO check dependency
        # tested with this on WSL with pymc v5.14.0
        # X_grid = pm.math.cartesian(x_cov[:, None], x_time[None, :])
        X_grid = pm.math.cartesian(x_time, x_cov)

        mu, _ = self.gp.predict(
            X_grid,
            point=self.mp,
            diag=True,
            pred_noise=True,
            model=self.model,
            )

        target = self.dm.y_t(mu)
        # TODO return a Dataset with the correct shape
        target = target.data.reshape(n_time, n_cov)
        index = self.dm.covariate_pipelines["time"].inverse_transform(x_time.reshape(-1, 1))
        covariate = self.dm.covariate_pipelines[covariate].inverse_transform(x_cov.reshape(-1, 1))

        return target, index, covariate

    @is_fitted
    def sample(self,
               covariates,
               n=1000,
               diag=False,
               pred_noise=False,
               method="cholesky",
               tol=1e-6,
               ) -> DataArray:
        """Sample from the posterior distribution of the model.

        Parameters
        ----------
        covariates : Dataset
            Covariates for prediction.
        n : int, optional
            Number of samples to draw.
        method{ ‘svd’, ‘eigh’, ‘cholesky’}, optional
            Method to use for covariance matrix decomposition. The default is
            ‘cholesky’.
        tol : float, optional
            Tolerance when checking the singular values in covariance matrix.
            The default is 1e-6.
        """
        Xnew = self.dm.Xnew(covariates)
        mu, cov = self.gp.predict(
            Xnew,
            point=self.mp,
            diag=diag,
            pred_noise=pred_noise,
            model=self.model,
        )

        rng = np.random.default_rng()
        sim = rng.multivariate_normal(mu, cov, size=n, method=method, tol=tol)

        # TODO modify transform to handle samples/draws HACK
        temp = self.dm.y_t(sim)
        data = temp.data.reshape(n, -1)
        attrs = temp.attrs
        da = DataArray(
            data,
            coords={"time": covariates.time, "draw": np.arange(n)},
            dims=["draw", "time"],
            attrs=attrs,
        )

        return da

    def build_model(self, X, y, **kwargs):
        """
        Creates an instance of pm.Model based on provided data and
        model_config, and attaches it to self.

        The subclass method must instantiate self.model and self.gp.

        Raises
        ------
        NotImplementedError
        """
        self.model = None
        self.gp = None

        raise NotImplementedError(
            "This method must be implemented in a subclass"
            )
