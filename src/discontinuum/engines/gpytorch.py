"""Data transformations to improve optimization"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gpytorch
import numpy as np
import torch
import tqdm
from xarray import DataArray

from discontinuum.engines.base import BaseModel, is_fitted

if TYPE_CHECKING:
    from typing import Dict, Optional
    from xarray import Dataset


class LatentGPyTorch(BaseModel):
    def __init__(
        self,
        model_config: Optional[Dict] = None,
    ):
        """ """
        super().__init__(model_config=model_config)

    def fit(self, covariates, target=None):
        pass


class MarginalGPyTorch(BaseModel):
    def __init__(
        self,
        model_config: Optional[Dict] = None,
    ):
        """ """
        super(BaseModel, self).__init__(model_config=model_config)

    def fit(
            self,
            covariates: Dataset,
            target: Dataset,
            iterations: int = 100,
            optimizer: str = "adam",
            ):
        """Fit the model to data.

        Parameters
        ----------
        covariates : Dataset
            Covariates for training.
        target : Dataset
            Target data for training.
        iterations : int, optional
            Number of iterations for optimization. The default is 100.
        optimizer : str, optional
            Optimization method. The default is "adam".
        """
        self.is_fitted = True
        # setup data manager (self.dm)
        self.dm.fit(target=target, covariates=covariates)

        self.X = self.dm.X
        self.y = self.dm.y
        train_x = torch.tensor(self.X, dtype=torch.float32)
        train_y = torch.tensor(self.y, dtype=torch.float32)

        self.model = self.build_model(train_x, train_y)
        # also sets self.likelihood

        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        else:
            raise NotImplementedError("Only Adam optimizer is implemented")

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        pbar = tqdm.tqdm(range(iterations), ncols=70)
        for i in pbar:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            pbar.set_postfix(loss=loss.item())
            optimizer.step()

    @is_fitted
    def predict(self,
                covariates: Dataset,
                diag=True,
                pred_noise=False,
                ) -> DataArray:
        """Uses the fitted model to make predictions on new data.

        Parameters
        ----------
        covariates : Dataset
            Covariates for prediction.
        diag : bool, optional
            Return only the diagonal of the covariance matrix.
            The default is True.
        pred_noise : bool, optional
            Include measurement uncertainty in the prediction.
            The default is False.
        """
        Xnew = torch.tensor(
            self.dm.Xnew(covariates),
            dtype=torch.float32,
        )

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(Xnew))
            mu = observed_pred.mean
            var = observed_pred.variance

        target = self.dm.y_t(mu)
        # TODO the reshape should be done in the pipeline
        se = self.dm.error_pipeline.inverse_transform(var.reshape(-1, 1))

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

        x_time = torch.linspace(x_min[time_dim], x_max[time_dim], n_time)
        x_cov = torch.linspace(x_min[covariate_dim], x_max[covariate_dim], n_cov)

        X_grid = torch.cartesian_prod(x_time, x_cov)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_grid))
            mu = observed_pred.mean
            # var = observed_pred.variance

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
               #diag=False,
               #pred_noise=False,
               #method="cholesky",
               #tol=1e-6,
               ) -> DataArray:
        """Sample from the posterior distribution of the model.

        Parameters
        ----------
        covariates : Dataset
            Covariates for prediction.
        n : int, optional
            Number of samples to draw.
        """
        Xnew = torch.tensor(
            self.dm.Xnew(covariates),
            dtype=torch.float32,
        )

        self.model.eval()
        self.likelihood.eval()

        f_preds = self.model(Xnew)

        # GPyTorch has several sampling optimizations, but none are working for me
        sim = f_preds.sample(sample_shape=torch.Size([n]))

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

    def build_model(self, X, y, **kwargs) -> gpytorch.models.ExactGP:
        """
        Creates an instance of pm.Model based on provided data and
        model_config, and attaches it to self.

        The subclass method must instantiate self.model and self.likelihood.

        Raises
        ------
        NotImplementedError
        """
        self.model = None

        raise NotImplementedError(
            "This method must be implemented in a subclass"
            )
