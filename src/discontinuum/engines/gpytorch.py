"""Data transformations to improve optimization"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gpytorch
import numpy as np
import torch
import tqdm
import xarray as xr
from xarray import DataArray, Dataset

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
        super().__init__(model_config=model_config)

    @classmethod
    def load(cls, ds: Dataset) -> MarginalGPyTorch:
        """Load a MarginalGPyTorch model from an xarray Dataset."""
        # Dispatch to concrete subclass if base class invoked
        model_class_path = ds.attrs.get('model_class')
        if model_class_path:
            # If called on base class, redirect
            full_name = f"{cls.__module__}.{cls.__name__}"
            if model_class_path != full_name:
                # dynamic import of subclass
                import importlib
                module_name, class_name = model_class_path.rsplit('.', 1)
                submod = importlib.import_module(module_name)
                subcls = getattr(submod, class_name)
                return subcls.load(ds)

        # Extract training data using stored variable names
        info = ds.attrs.get('model', {})
        target = ds[ info['target'] ]
        covariates = ds[ info['covariates'] ]
        target_unc = ds[ info['target_unc'] ] if info.get('target_unc') else None

        # Instantiate and fit model structure
        model = cls()
        model.fit(covariates=covariates, target=target, target_unc=target_unc, iterations=1)

        # Restore model parameters
        state = info.get('state_dict')
        if state is not None:
            model.model.load_state_dict(state)
        model.is_fitted = True
        return model

    def save(self) -> Dataset:
        """Save the model's training data and parameters to an xarray Dataset."""
        # Save training data as xarray Dataset
        datasets = [
            self.dm.data.target.to_dataset(),
            self.dm.data.covariates,
            ]
        if self.dm.data.target_unc is not None:
            datasets.append(self.dm.data.target_unc)

        ds = xr.merge(datasets)

        #ds.attrs['state_dict'] = self.model.state_dict()
        ds.attrs["model"] = {
            "state_dict": self.model.state_dict(),
            "covariates": list(self.dm.data.covariates.data_vars.keys()),
            "target": self.dm.data.target.name,
            "target_unc": self.dm.data.target_unc.name if self.dm.data.target_unc is not None else None,
        }

        return ds

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
        """
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
            # More conservative starting LR
            learning_rate = 0.05
        
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
            factor=0.7,                      # Reduce LR more gradually (30% reduction)
            patience=max(20, patience // 2), # Scheduler should be less patient than early stopping
            threshold=1e-4,                  # Less sensitive threshold (0.01% improvement)
            threshold_mode='rel',            # Use relative threshold
            min_lr=1e-6
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training loop with stability features
        pbar = tqdm.tqdm(range(iterations), ncols=100)  # Wider progress bar
        best_loss = float('inf')
        patience_counter = 0
        min_improvement = 1e-6  # Minimum improvement to reset patience counter
        

        nan_loss_counter = 0
        try:
            for i in pbar:
                optimizer_obj.zero_grad()
                output = self.model(train_x)
                
                try:
                    loss = -mll(output, train_y)
                except Exception as e:
                    nan_loss_counter += 1
                    if nan_loss_counter > 10:
                        raise e
                    continue
                
                # Check for NaN/Inf loss after successful computation
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_loss_counter += 1
                    if nan_loss_counter > 10:
                        raise RuntimeError(f"Encountered more than 10 consecutive NaN/Inf losses at iteration {i+1}")
                    continue
                    
                nan_loss_counter = 0

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Check for NaN gradients
                has_nan_grad = any(
                    param.grad is not None and torch.isnan(param.grad).any()
                    for param in self.model.parameters()
                )

                if has_nan_grad:
                    # Update best loss tracking (loss is still valid, just gradients are NaN)
                    if loss.item() < best_loss - min_improvement:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    current_lr = optimizer_obj.param_groups[0]['lr']
                    pbar.set_postfix_str(
                        f'loss={loss.item():.4f} lr={current_lr:.1e} | NaN gradients - skipping step'
                    )
                    continue

                optimizer_obj.step()

                # Update learning rate scheduler
                if early_stopping:
                    scheduler.step(loss.item())

                # Early stopping check
                if loss.item() < best_loss - min_improvement:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Update progress bar
                current_lr = optimizer_obj.param_groups[0]['lr']
                progress_info = f'loss={loss.item():.4f} lr={current_lr:.1e}'
                pbar.set_postfix_str(progress_info)

                if early_stopping and patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {i+1} iterations")
                    print(f"Best loss: {best_loss:.6f}")
                    break

        except KeyboardInterrupt:
            print(f"\nTraining interrupted at iteration {i+1}")
            print(f"Best loss: {best_loss:.6f}")
        finally:
            # Mark as fitted after any training iterations have completed
            self.is_fitted = True

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
