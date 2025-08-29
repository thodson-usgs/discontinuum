import gpytorch
import numpy as np
import torch
from discontinuum.engines.gpytorch import MarginalGPyTorch, NoOpMean


from gpytorch.kernels import (
    MaternKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
    PeriodicKernel,
)
from gpytorch.priors import (
    GammaPrior,
    HalfNormalPrior,
    HalfCauchyPrior,
    NormalPrior,
)

from rating_gp.models.base import RatingDataMixin, ModelConfig
from rating_gp.plot import RatingPlotMixin
from rating_gp.models.kernels import (
    SigmoidKernel,
    InvertedSigmoidKernel,
    LogWarpKernel,
)


class PowerLawTransform(torch.nn.Module):
    """
    """
    def __init__(self):
        super(PowerLawTransform, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(1))
        # initialize b according to Manning's equation (refer to clamp)
        # Use proper parameter initialization to maintain gradient flow
        self.b = torch.nn.Parameter(torch.randn(1) + 1.3)
        # stage is scaled to 1-2, so initialize c to 0-1
        self.c = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        self.c.data = torch.clamp(self.c.data, max=x.min()-1e-6)
        return self.a + (self.b * torch.log(x - self.c))


class RatingGPMarginalGPyTorch(
    RatingDataMixin,
    RatingPlotMixin,
    # comes last b/c it conflics with Mixin
    MarginalGPyTorch,
):
    """
    Gaussian Process implementation of the LOAD ESTimation (LOADEST) model

    This model currrently uses the marginal likelihood implementation, which is
    fast but does not account for censored data. Censored data require a slower
    latent variable implementation.
    """
    def __init__(
            self,
            model_config: ModelConfig = ModelConfig(),
    ):
        """ """
        # Ensure MarginalGPyTorch.__init__ executes to set checkpoint fields
        super().__init__(model_config=model_config)
        self.build_datamanager(model_config)
        

    def build_model(self, X, y, y_unc=None) -> gpytorch.models.ExactGP:
        """Build marginal likelihood version of RatingGP
        """
        # assume a constant measurement error for testing
        if y_unc is not None:
            noise = y_unc
        else:
            noise = 0.1**2 * torch.ones(y.shape[0]).reshape(1, -1)
        # TODO: Fix "GPInputWarning: You have passed data through a 
        # FixedNoiseGaussianLikelihood that did not match the size of the fixed
        # noise, *and* you did not specify noise. This is treated as a no-op."
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise,
            #learn_additional_noise=False,
            learn_additional_noise=True,
            noise_prior=gpytorch.priors.HalfNormalPrior(scale=0.03),
        )

        model = ExactGPModel(X, y, self.likelihood)

        return model

    def fit(self, covariates, target, target_unc=None, iterations=100, optimizer=None,
            learning_rate=None, early_stopping=False, patience=60, scheduler=True,
            resume=False, monotonic_penalty_weight: float = 0.0, grid_size: int = 64,
            monotonic_penalty_interval: int = 1):
        """
        Override fit to inject a monotonicity penalty on the rating curve.

        resume: if True, continue training from the last saved iteration until the total
          number of iterations is reached. If False, start from iteration 0.
        monotonic_penalty_weight: strength of penalty on negative dQ/dStage.
          1.0 works well in practice.
        grid_size: number of random points to sample over time-stage grid
        monotonic_penalty_interval: compute the penalty every k iterations (k>=1).
          When k>1, the penalty is applied every k-th iteration and scaled by k
          to maintain the same expected regularization strength, reducing compute.
        """
        # If no penalty, just call base implementation
        if monotonic_penalty_weight <= 0:
            return super().fit(
                covariates=covariates,
                target=target,
                target_unc=target_unc,
                iterations=iterations,
                optimizer=optimizer,
                learning_rate=learning_rate,
                early_stopping=early_stopping,
                patience=patience,
                scheduler=scheduler,
                resume=resume,
                penalty_callback=None,
                penalty_weight=0.0,
            )

        # Simple counter to optionally skip penalty some iterations for speed
        step = {"i": 0}

        # Define penalty callback that depends on current model params
        def penalty_callback():
            # Optionally skip penalty to save compute
            step["i"] += 1
            if monotonic_penalty_interval > 1 and (step["i"] % monotonic_penalty_interval) != 0:
                # Return a 0 scalar on the correct device (no gradient contribution this step)
                dev = next(self.model.parameters()).device
                return torch.zeros((), device=dev)

            # Sample a random grid in transformed model space
            # time is uniform, stage is log-uniform (denser at low values)
            time_dim = 0
            stage_dim = 1

            X_np = self.dm.X  # training transformed inputs
            x_min = X_np.min(axis=0)
            x_max = X_np.max(axis=0)
            device = next(self.model.parameters()).device
            # Uniform for time
            u_time = torch.rand((grid_size,), dtype=torch.float32, device=device)
            time_grid = u_time * (x_max[time_dim] - x_min[time_dim]) + x_min[time_dim]
            # Log-uniform for stage, add epsilon to avoid log(0)
            eps = 1e-6
            log_xmin = float(np.log(x_min[stage_dim] + eps))
            log_xmax = float(np.log(x_max[stage_dim] + eps))
            u_stage = torch.rand((grid_size,), dtype=torch.float32, device=device)
            log_stage_grid = u_stage * (log_xmax - log_xmin) + log_xmin
            # Only the stage dimension requires gradients
            stage_grid = torch.exp(log_stage_grid).requires_grad_(True)
            # Stack into grid
            x_grid = torch.stack([time_grid, stage_grid], dim=1)

            was_model_training = self.model.training
            was_likelihood_training = self.likelihood.training
            self.model.eval()
            self.likelihood.eval()
            try:
                with gpytorch.settings.fast_pred_var():
                    mean = self.likelihood(self.model(x_grid)).mean
            finally:
                if was_model_training:
                    self.model.train()
                if was_likelihood_training:
                    self.likelihood.train()

            d_mean_d_stage = torch.autograd.grad(mean.sum(), stage_grid, create_graph=True)[0]
            neg = torch.clamp(-d_mean_d_stage, min=0.0)
            pen = neg.mean()
            if monotonic_penalty_interval > 1:
                pen = pen * float(monotonic_penalty_interval)
            return pen

        return super().fit(
            covariates=covariates,
            target=target,
            target_unc=target_unc,
            iterations=iterations,
            optimizer=optimizer,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            patience=patience,
            scheduler=scheduler,
            resume=resume,
            penalty_callback=penalty_callback,
            penalty_weight=float(monotonic_penalty_weight),
        )


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        n_d = train_x.shape[1]  # number of dimensions
        assert n_d == 2, "Only two dimensions supported"

        self.dims = np.arange(n_d)
        self.time_dim = [self.dims[0]]
        self.stage_dim = [self.dims[1]]

        self.powerlaw = PowerLawTransform()

        self.mean_module = NoOpMean()

        # Use stage (not y) for sigmoid kernel constraint
        stage = train_x[:, self.stage_dim[0]]#.cpu().numpy()
        b_min = np.quantile(stage, 0.10)
        b_max = np.quantile(stage, 0.90)
 
        # Create sigmoid kernel for gating (shared switchpoint)
        sigmoid_lower = SigmoidKernel(
            active_dims=self.stage_dim,
            b_constraint=gpytorch.constraints.Interval(b_min, b_max),
        )
 
        sigmoid_upper = InvertedSigmoidKernel(
            sigmoid_kernel=sigmoid_lower,
            active_dims=self.stage_dim,
            b_constraint=gpytorch.constraints.Interval(b_min, b_max),
        )
 
        # Compose the upper kernel branch and wrap in LogWarpKernel
        kernel = (
            self.cov_base(eta_prior=HalfNormalPrior(scale=1.0))
            +
            self.cov_periodic(eta_prior=HalfNormalPrior(scale=0.4))
        )

        upper_kernel = (
            self.cov_bend(eta_prior=HalfNormalPrior(scale=0.6))
        )

        lower_kernel = (
            self.cov_shift(
                eta_prior=HalfNormalPrior(scale=0.6),
                time_prior=GammaPrior(concentration=1, rate=30),
            )
        )

        upper_kernel_warped = LogWarpKernel(upper_kernel, self.stage_dim[0])
        lower_kernel_warped = LogWarpKernel(lower_kernel, self.stage_dim[0])
        kernel_warped = LogWarpKernel(kernel, self.stage_dim[0])

        self.covar_module = (
            sigmoid_lower * lower_kernel_warped
            +
            sigmoid_upper * upper_kernel_warped
            +
            kernel_warped
        )


    def forward(self, x):
        self.powerlaw.b.data.clamp_(1.2, 2.5)
        #x = x.clone()
        #q = self.powerlaw(x[:, self.stage_dim])
        #x_t[:, self.stage_dim] = self.warp_stage_dim(x_t[:, self.stage_dim])
        x_t = x.clone()
        x_t[:, self.stage_dim[0]] = self.powerlaw(x_t[:, self.stage_dim[0]])
        q = x_t[:, self.stage_dim[0]]
        mean_x = self.mean_module(q)

        #covar_x = self.covar_module(x_t)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def cov_stage(self, ls_prior=None):
        eta = HalfNormalPrior(scale=1)
        
        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=ls_prior,
                nu=2.5,
            ),
            outputscale_prior=eta,
        )

    def cov_time(self, ls_prior=None, eta_prior=None):
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=1)

        # Base Matern kernel for long-term trends
        return ScaleKernel(
            MaternKernel(
                active_dims=self.time_dim,
                lengthscale_prior=ls_prior,
                nu=1.5,
            ),
            outputscale_prior=eta_prior,
        )
    
    def cov_shift(self, eta_prior=None, time_prior=None):
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=0.3) 

        if time_prior is None:
            time_prior = GammaPrior(concentration=1, rate=7)

        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=GammaPrior(concentration=1., rate=1.),
                nu=2.5,
            ) 
            *
            MaternKernel(
                active_dims=self.time_dim,
                lengthscale_prior=time_prior,
                nu=2.5,
            ),
            outputscale_prior=eta_prior,
        )
 
    
    def cov_bend(self, eta_prior=None):
        """
        Smooth, time-dependent bending kernel for switchpoint.
        """
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=0.2) 

        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                #lengthscale_prior=GammaPrior(concentration=2, rate=1),
                lengthscale_prior=GammaPrior(concentration=3, rate=2),
            ) 
            *
            MaternKernel(
                active_dims=self.time_dim,
                lengthscale_prior=GammaPrior(concentration=4, rate=2),
            ),
            outputscale_prior=eta_prior,
        )
    
    def cov_periodic(self, ls_prior=None, eta_prior=None):
        """
        Smooth, time-dependent periodic kernel for seasonal effects.
        """
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=0.5) 

        if ls_prior is None:
            #ls_prior = GammaPrior(concentration=2, rate=4)
            ls_prior = GammaPrior(concentration=9, rate=10)


        return ScaleKernel(
            PeriodicKernel(
                active_dims=self.time_dim,
                period_length_prior=NormalPrior(loc=1.0, scale=0.05),  # ~1 year
                lengthscale_prior=ls_prior,
            ),
            outputscale_prior=eta_prior,
        )
    
    def cov_base(self, eta_prior=None):
        """
        Smooth, time-independent base rating curve using a Matern kernel on stage.
        """
        if eta_prior is None:
            eta = HalfNormalPrior(scale=1.0)
        else:
            eta = eta_prior

        #ls = GammaPrior(concentration=3, rate=1)
        ls = GammaPrior(concentration=4., rate=4.)
        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )
    