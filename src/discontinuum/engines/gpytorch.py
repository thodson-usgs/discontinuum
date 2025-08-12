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
    from typing import Dict, Optional, Tuple, Callable
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
        # --- Checkpointing state ---
        self._resume_info = None  # holds checkpoint info to resume training
        self._last_optimizer = None
        self._last_scheduler = None

    @classmethod
    def load(
        cls,
        filepath: str,
        covariates: Dataset,
        target: Dataset,
        target_unc: Optional[Dataset] = None,
    ) -> 'MarginalGPyTorch':
        """Load a model from a checkpoint file and initialize for resume.

        Parameters
        ----------
        filepath : str
            Path to a checkpoint saved by save().
        covariates : xarray.Dataset
            Training covariates used to rebuild the data manager.
        target : xarray.DataArray or Dataset
            Training target used to rebuild the data manager.
        target_unc : xarray.DataArray or Dataset, optional
            Training uncertainty used to rebuild the data manager.
        """
        ckpt = torch.load(filepath, map_location='cpu')

        model = cls()
        # Setup data manager and tensors
        model.dm.fit(target=target, covariates=covariates, target_unc=target_unc)
        model.X = model.dm.X
        model.y = model.dm.y
        train_x = torch.tensor(model.X, dtype=torch.float32)
        train_y = torch.tensor(model.y, dtype=torch.float32)

        if target_unc is None:
            model.model = model.build_model(train_x, train_y)
        else:
            model.y_unc = model.dm.y_unc
            train_y_unc = torch.tensor(model.y_unc, dtype=torch.float32)
            model.model = model.build_model(train_x, train_y, train_y_unc)

        # Restore model/likelihood
        model.model.load_state_dict(ckpt['model_state_dict'])
        if 'likelihood_state_dict' in ckpt and ckpt['likelihood_state_dict'] is not None:
            model.likelihood.load_state_dict(ckpt['likelihood_state_dict'])

        # Store optimizer/scheduler info to apply in fit()
        model._resume_info = {
            'optimizer_state_dict': ckpt.get('optimizer_state_dict'),
            'optimizer_name': ckpt.get('optimizer_name'),
            'optimizer_lr': ckpt.get('optimizer_lr'),
            'scheduler_state_dict': ckpt.get('scheduler_state_dict'),
            'scheduler_name': ckpt.get('scheduler_name'),
        }

        # Mark as fitted (weights are loaded), but allow further training
        model.is_fitted = True
        return model

    def save(
        self,
        filepath: str,
        optimizer_obj: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """Save the model to a checkpoint file (weights + optimizer/scheduler).

        Parameters
        ----------
        filepath : str
            Path to save the checkpoint (.pt/.pth).
        optimizer_obj : torch.optim.Optimizer, optional
            Optimizer whose state will be saved. If None, uses the last one from fit().
        scheduler : torch.optim.lr_scheduler, optional
            LR scheduler whose state will be saved. If None, uses the last one from fit().
        extra : dict, optional
            Extra metadata to include in the checkpoint.
        """
        # Prefer last-used optimizer/scheduler if not provided explicitly
        if optimizer_obj is None:
            optimizer_obj = getattr(self, '_last_optimizer', None)
        if scheduler is None:
            scheduler = getattr(self, '_last_scheduler', None)

        if not hasattr(self, 'model'):
            raise RuntimeError('No model to save. Call fit() first.')
        if not hasattr(self, 'likelihood'):
            raise RuntimeError('No likelihood to save. Call fit() first.')

        # Normalize optimizer name for easy restoration
        opt_name = None
        lr_val = None
        if optimizer_obj is not None:
            if isinstance(optimizer_obj, torch.optim.AdamW):
                opt_name = 'adamw'
            elif isinstance(optimizer_obj, torch.optim.Adam):
                opt_name = 'adam'
            else:
                opt_name = optimizer_obj.__class__.__name__
            # capture first param group LR as hint
            try:
                lr_val = optimizer_obj.param_groups[0].get('lr', None)
            except Exception:
                lr_val = None

        ckpt = {
            'model_class': f"{self.__class__.__module__}.{self.__class__.__name__}",
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'optimizer_state_dict': optimizer_obj.state_dict() if optimizer_obj is not None else None,
            'optimizer_name': opt_name,
            'optimizer_lr': lr_val,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scheduler_name': scheduler.__class__.__name__ if scheduler is not None else None,
            'model_config': getattr(self, 'model_config', None),
            'extra': extra or {},
        }
        torch.save(ckpt, filepath)

    def fit(
            self,
            covariates: Dataset,
            target: Dataset,
            target_unc: Dataset = None,
            iterations: int = 100,
            optimizer: Optional[str] = None,
            learning_rate: float = None,
            early_stopping: bool = False,
            patience: int = 60,
            scheduler: bool = True,
            penalty_callback: 'Optional[Callable[[], torch.Tensor]]' = None,
            penalty_weight: float = 0.0,
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
            Optimization method. Supported: "adam", "adamw". If None, uses the
            optimizer stored in a loaded checkpoint, otherwise defaults to "adam".
        learning_rate : float, optional
            Learning rate for optimization. If None, uses the value from checkpoint
            if available, otherwise a conservative default.
        early_stopping : bool, optional
            Whether to use early stopping. The default is False.
        patience : int, optional
            Number of iterations to wait without improvement before stopping. The default is 60.
        scheduler : bool, optional
            Whether to use a learning rate scheduler. The default is True.
        penalty_callback : callable, optional
            A zero-argument callable that returns a scalar torch.Tensor penalty to be added
            to the optimization objective each iteration. This enables model-specific
            regularization such as monotonicity constraints.
        penalty_weight : float, optional
            Multiplier applied to the penalty value when added to the objective. Default 0.0.
        """
        # setup data manager (self.dm)
        self.dm.fit(target=target, covariates=covariates, target_unc=target_unc)

        self.X = self.dm.X
        self.y = self.dm.y
        train_x = torch.tensor(self.X, dtype=torch.float32)
        train_y = torch.tensor(self.y, dtype=torch.float32)

        resuming = self._resume_info is not None and getattr(self, 'model', None) is not None and getattr(self, 'likelihood', None) is not None

        if not resuming:
            # fresh build
            if target_unc is None:
                self.model = self.build_model(train_x, train_y)
                # also sets self.likelihood
            else:
                self.y_unc = self.dm.y_unc
                train_y_unc = torch.tensor(self.y_unc, dtype=torch.float32)
                self.model = self.build_model(train_x, train_y, train_y_unc)
        else:
            # Continue training with existing model/likelihood; update training data if needed
            try:
                self.model.set_train_data(inputs=train_x, targets=train_y, strict=False)
            except Exception:
                # If that fails (e.g., structure changed), rebuild
                if target_unc is None:
                    self.model = self.build_model(train_x, train_y)
                else:
                    self.y_unc = self.dm.y_unc
                    train_y_unc = torch.tensor(self.y_unc, dtype=torch.float32)
                    self.model = self.build_model(train_x, train_y, train_y_unc)
                resuming = False  # cannot restore optimizer state safely

        self.model.train()
        self.likelihood.train()

        # If resuming, prefer saved optimizer settings when caller uses defaults
        resume = self._resume_info or {}
        opt_name_saved = resume.get('optimizer_name')
        lr_saved = resume.get('optimizer_lr')
        opt_choice = optimizer if optimizer is not None else (opt_name_saved or 'adam')
        lr_choice = learning_rate if learning_rate is not None else (lr_saved or 0.05)

        if opt_choice == "adamw" or (opt_name_saved and opt_name_saved.lower() == 'adamw'):
            optimizer_obj = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr_choice,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-2,
            )
        elif opt_choice == "adam" or (opt_name_saved and opt_name_saved.lower() == 'adam'):
            optimizer_obj = torch.optim.Adam(
                self.model.parameters(),
                lr=lr_choice,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-4,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {opt_choice!r}. Supported optimizers are 'adam' and 'adamw'."
            )

        # Restore optimizer state if resuming and compatible
        if resuming and resume.get('optimizer_state_dict') is not None:
            try:
                optimizer_obj.load_state_dict(resume['optimizer_state_dict'])
            except Exception:
                pass

        if scheduler:
            scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_obj,
                mode='min',
                factor=0.7,
                patience=max(20, patience // 2),
                threshold=1e-4,
                threshold_mode='rel',
                min_lr=1e-6,
                cooldown=10,
            )
            # Restore scheduler if resuming
            if resuming and resume.get('scheduler_state_dict') is not None:
                try:
                    scheduler_obj.load_state_dict(resume['scheduler_state_dict'])
                except Exception:
                    pass
        else:
            scheduler_obj = None

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training loop with stability features
        pbar = tqdm.tqdm(range(iterations), ncols=100)
        best_obj = float('inf')
        patience_counter = 0
        min_improvement = 1e-6
        
        nan_loss_counter = 0
        try:
            for i in pbar:
                optimizer_obj.zero_grad(set_to_none=True)
                output = self.model(train_x)
                
                try:
                    nll = -mll(output, train_y)
                except Exception as e:
                    nan_loss_counter += 1
                    if nan_loss_counter > 10:
                        raise e
                    continue
                
                # Optional penalty term
                penalty_val = None
                if penalty_callback is not None and penalty_weight > 0.0:
                    try:
                        penalty_val = penalty_callback()
                        # basic sanity check
                        if not torch.is_tensor(penalty_val):
                            penalty_val = None
                    except Exception:
                        penalty_val = None
                
                objective = nll
                if penalty_val is not None:
                    objective = objective + float(penalty_weight) * penalty_val
                
                # Check for NaN/Inf objective after successful computation
                if torch.isnan(objective) or torch.isinf(objective):
                    nan_loss_counter += 1
                    if nan_loss_counter > 10:
                        raise RuntimeError(f"Encountered more than 10 consecutive NaN/Inf objectives at iteration {i+1}")
                    continue
                    
                nan_loss_counter = 0

                objective.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Check for NaN gradients
                has_nan_grad = any(
                    param.grad is not None and torch.isnan(param.grad).any()
                    for param in self.model.parameters()
                )

                if has_nan_grad:
                    # Update best objective tracking (objective is still valid, just gradients are NaN)
                    obj_item = float(objective.item())
                    if obj_item < best_obj - min_improvement:
                        best_obj = obj_item
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Sanitize NaN/Inf gradients and update parameters to allow loss change
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    optimizer_obj.step()
                    if scheduler_obj is not None:
                        scheduler_obj.step(obj_item)

                    # Early stopping logic (independent of scheduler)
                    if obj_item < best_obj - min_improvement:
                        best_obj = obj_item
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    current_lr = optimizer_obj.param_groups[0]['lr']
                    if early_stopping and patience_counter >= patience:
                        print(f"\nEarly stopping triggered after {i+1} iterations")
                        print(f"Best objective: {best_obj:.6f}")
                        break
                    continue

                optimizer_obj.step()
                obj_item = float(objective.item())
                if scheduler_obj is not None:
                    scheduler_obj.step(obj_item)

                # Early stopping check
                if obj_item < best_obj - min_improvement:
                    best_obj = obj_item
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Update progress bar
                current_lr = optimizer_obj.param_groups[0]['lr']
                suffix = f'obj={obj_item:.4f}, lr={current_lr:.1e}'
                if penalty_val is not None:
                    try:
                        suffix += f', pen={float(penalty_val.item()):.3e}'
                    except Exception:
                        pass
                if nan_loss_counter > 0:
                    suffix += f' | NaN gradients: {nan_loss_counter}'
                pbar.set_postfix_str(suffix)

                if early_stopping and patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {i+1} iterations")
                    print(f"Best objective: {best_obj:.6f}")
                    break

        except KeyboardInterrupt:
            print(f"\nTraining interrupted at iteration {i+1}")
            print(f"Best objective: {best_obj:.6f}")
        finally:
            # Mark as fitted after any training iterations have completed
            self.is_fitted = True

        # Optionally: keep last optimizer/scheduler for quick checkpointing by caller
        self._last_optimizer = optimizer_obj
        self._last_scheduler = scheduler_obj

        # Return for chaining
        return None

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
