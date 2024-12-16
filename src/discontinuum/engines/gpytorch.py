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
    from numpy.typing import ArrayLike
    from typing import Dict, Optional, Tuple
    from xarray import Dataset


class NoOpMean(gpytorch.means.Mean):
    def forward(self, x):
        return x.squeeze(-1)


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
            target_unc: Dataset = None,
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
        target_unc : Dataset, optional
            Uncertainty on target data.
        iterations : int, optional
            Number of iterations for optimization. The default is 100.
        optimizer : str, optional
            Optimization method. The default is "adam".
        """
        self.is_fitted = True
        # setup data manager (self.dm)
        self.dm.fit(target=target, covariates=covariates, target_unc=target_unc)

        self.X = self.dm.X
        self.y = self.dm.y
        train_x = torch.tensor(self.X, dtype=torch.float32)
        train_y = torch.tensor(self.y, dtype=torch.float32)

        if target_unc is None:
            self.model = self.build_model(train_x, train_y)
            # also sets self.likelihood
        else:
            self.y_unc = self.dm.y_unc
            train_y_unc = torch.tensor(self.y_unc, dtype=torch.float32)
            self.model = self.build_model(train_x, train_y, train_y_unc)

        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05) # default previously lr=0.1
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
                ) -> Tuple[DataArray, DataArray]:
        """Uses the fitted model to make predictions on new data.

        The input and output are in the original data space.

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

        Returns
        -------
        target : DataArray
            Target prediction.
        se : DataArray
            Standard error of the prediction.
        """
        Xnew = torch.tensor(
            self.dm.Xnew(covariates),
            dtype=torch.float32,
        )

        mu, var = self.__gpytorch_predict(Xnew)

        target = self.dm.y_t(mu)
        target = target.assign_coords(covariates.coords)
        se = self.dm.error_pipeline.inverse_transform(var)
        se = se.assign_coords(covariates.coords)

        return target, se

    @is_fitted
    def predict_grid(self,
                     covariate: str,
                     coord: str = None,
                     t_step: int = 12):
        """Predict on a grid of points.

        Parameters
        ----------
        covariate : str
            Covariate dimension to predict on.
        coord : str, optional
            Coordinate of the covariate dimension to predict on. The default
            is the first coordinate of the covariate.
        t_step : int, optional
            Number of grid points per step in coord units. The default is 12.
        """
        if coord is None:
            coord = list(self.dm.data.covariates.coords)[0]
        coord_dim = self.dm.get_dim(coord)
        covariate_dim = self.dm.get_dim(covariate)

        x_max = self.dm.X.max(axis=0)
        x_min = self.dm.X.min(axis=0)
        x_range = x_max - x_min

        n_cov = 18
        n_coord = np.round(x_range[coord_dim] * t_step).astype(int)

        x_coord = torch.linspace(x_min[coord_dim], x_max[coord_dim], n_coord)
        x_cov = torch.linspace(x_min[covariate_dim], x_max[covariate_dim], n_cov)

        # expects a 1D vector
        X_grid = torch.cartesian_prod(x_coord, x_cov)

        mu, var = self.__gpytorch_predict(X_grid)

        target = self.dm.y_t(mu)

        # TODO handle type conversion in pipeline
        index = self.dm.covariate_pipelines[coord].inverse_transform(x_coord.numpy())
        covariates = self.dm.covariate_pipelines[covariate].inverse_transform(x_cov.numpy())

        da = DataArray(
            target.data.reshape(n_coord, n_cov),
            coords=[index, covariates],
            dims=[coord, covariate],
            attrs=target.attrs,
        )

        return da

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

        sim = f_preds.sample(sample_shape=torch.Size([n]))

        # TODO modify transform to handle draws
        # flatten then reshape to work around our transformation pipeline
        temp = self.dm.y_t(sim.flatten())
        data = temp.data.reshape(n, -1)
        attrs = temp.attrs
        da = DataArray(
            data,
            coords=dict(covariates.coords, draw=np.arange(n)),
            dims=["draw"] + list(covariates.coords),
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

    @is_fitted
    def __gpytorch_predict(
            self,
            x: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model-space prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input data in the (transformed) model space.

        Returns
        -------
        mu : torch.Tensor
            Mean of the prediction in the model space.
        var : torch.Tensor
            Variance of the prediction in the model space.
        """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            mu = observed_pred.mean
            var = observed_pred.variance

        return mu, var
