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
            optimizer: str = "adamw",
            learning_rate: float = None,
            early_stopping: bool = False,
            patience: int = 60,
            gradient_noise: bool = False,
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
            Optimization method. Supported: "adam", "adamw". The default is "adamw".
        learning_rate : float, optional
            Learning rate for optimization. If None, uses adaptive defaults.
        early_stopping : bool, optional
            Whether to use early stopping. The default is False.
        patience : int, optional
            Number of iterations to wait without improvement before stopping. The default is 60.
        gradient_noise : bool, optional
            Whether to inject Gaussian noise into gradients each step (std = 0.1 × current learning rate). The default is False.
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

        if learning_rate is None:
            if optimizer == "adam":
                learning_rate = 0.1  # Aggressive default for faster convergence
            elif optimizer == "adamw":
                learning_rate = 0.1
        
        if optimizer == "adamw":
            optimizer_obj = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-2      # Stronger regularization for AdamW
            )
        elif optimizer == "adam":
            optimizer_obj = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-4      # Lighter regularization for Adam
            )
        else:
            raise NotImplementedError(f"Only 'adam' and 'adamw' optimizers are supported. Got '{optimizer}'.")

        # Use ReduceLROnPlateau for more stable learning rate adaptation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_obj,
            mode='min',
            factor=0.5,                      # Reduce LR by half
            patience=max(2, patience),
            threshold=1e-4,
            min_lr=1e-5
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training loop with stability features
        pbar = tqdm.tqdm(range(iterations), ncols=100)  # Wider progress bar
        jitter = 1e-6  # Dynamic jitter for numerical stability
        best_loss = float('inf')
        patience_counter = 0
        min_lr_for_early_stop = 2e-5  # Stop if patience is exceeded and LR is below this
        
        for i in pbar:
            # Adam/AdamW optimizer with stability features
            optimizer_obj.zero_grad()
            output = self.model(train_x)

            # Attempt loss calculation with dynamic jitter
            try:
                with gpytorch.settings.cholesky_jitter(jitter):
                    loss = -mll(output, train_y)
            except Exception as e:
                # Increase jitter if numerical issues occur
                jitter = min(jitter * 10, 1e-2)
                current_lr = optimizer_obj.param_groups[0]['lr']
                pbar.set_postfix_str(
                    f'lr={current_lr:.1e} jitter={jitter:.1e} | Numerical issue - increasing jitter'
                )
                continue

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                current_lr = optimizer_obj.param_groups[0]['lr']
                pbar.set_postfix_str(
                    f'lr={current_lr:.1e} jitter={jitter:.1e} | NaN/Inf loss detected - skipping step'
                )
                continue

            loss.backward()

            # Get current learning rate before gradient noise injection
            current_lr = optimizer_obj.param_groups[0]['lr']

            # Gradient noise injection (if enabled)
            if gradient_noise:
                gradient_noise_scale = 0.1
                adaptive_noise = gradient_noise_scale * current_lr
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(mean=0.0, std=adaptive_noise, size=param.grad.shape, device=param.grad.device)
                        param.grad.add_(noise)

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Check for NaN gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                # Don't update scheduler on NaN gradients - this prevents rapid LR decay
                # The scheduler should only respond to actual optimization progress
                current_lr = optimizer_obj.param_groups[0]['lr']

                # Update best loss tracking (loss is still valid, just gradients are NaN)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Display comprehensive info even with NaN gradients, skip normal progress update
                pbar.set_postfix_str(
                    f'loss={loss.item():.4f} lr={current_lr:.1e} jitter={jitter:.1e} best={best_loss:.4f} | NaN gradients - skipping step'
                )
                continue

            optimizer_obj.step()

            # Update learning rate scheduler for Adam/AdamW
            scheduler.step(loss.item())
            current_lr = optimizer_obj.param_groups[0]['lr']

            # Early stopping check (more aggressive)
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            # Only update progress bar if not skipped above
            if not has_nan_grad:
                progress_info = f'loss={loss.item():.4f} lr={current_lr:.1e} jitter={jitter:.1e} best={best_loss:.4f}'
                if early_stopping:
                    progress_info += f' patience={patience_counter}/{patience}'
                pbar.set_postfix_str(progress_info)

            if early_stopping and patience_counter >= patience and current_lr <= min_lr_for_early_stop:
                print(f"\nEarly stopping triggered after {i+1} iterations")
                print(f"Best loss: {best_loss:.6f}")
                break

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
