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
    NormalPrior,
)

from rating_gp.models.base import RatingDataMixin, ModelConfig
from rating_gp.plot import RatingPlotMixin
from rating_gp.models.kernels import SigmoidKernel, InvertedSigmoidKernel


# --- Custom kernel for warping input with PowerLawTransform ---
class PowerLawWarpKernel(gpytorch.kernels.Kernel):
    """
    Wraps a base kernel and applies a PowerLawTransform to the stage input.
    """
    def __init__(self, base_kernel, powerlaw_transform, stage_dim):
        super().__init__()
        self.base_kernel = base_kernel
        self.powerlaw_transform = powerlaw_transform
        self.stage_dim = stage_dim

    def forward(self, x1, x2=None, **params):
        x1_ = x1.clone()
        x1_[:, self.stage_dim] = self.powerlaw_transform(x1_[:, self.stage_dim])
        if x2 is not None:
            x2_ = x2.clone()
            x2_[:, self.stage_dim] = self.powerlaw_transform(x2_[:, self.stage_dim])
        else:
            x2_ = None
        return self.base_kernel(x1_, x2_, **params)


class PowerLawTransform(torch.nn.Module):
    """
    Power law transformation for rating curves.
    """
    def __init__(self):
        super(PowerLawTransform, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(1))
        # register b and constrain it to avoid in-place clamping elsewhere
        self.b = torch.nn.Parameter(torch.randn(1))
        self.register_parameter("b", self.b)
        # use a broad positive interval for b
        try:
            self.register_constraint("b", gpytorch.constraints.Interval(1e-6, 50.0))
        except Exception:
            # If gpytorch.Module constraint helpers aren't available in this
            # environment, fall back to leaving b unconstrained (rare).
            pass
        # stage is scaled to 1-2, so initialize c to 0-1
        self.c = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        # Do not mutate parameter tensors in-place. Compute a clamped view for
        # c and use the constrained view for b if available.
        min_x = float(x.min())
        clamped_c = torch.clamp(self.c, max=(min_x - 1e-6))
        # use constrained b if constraint was registered
        try:
            b_constrained = self.b_constraint.transform(self.b)
        except Exception:
            b_constrained = self.b
        return self.a + (b_constrained * torch.log(x - clamped_c))


class RatingGPVariationalGPyTorch(
    RatingDataMixin,
    RatingPlotMixin,
    MarginalGPyTorch,
):
    """
    Variational Gaussian Process implementation of the LOAD ESTimation (LOADEST) model
    using sparse GPs with inducing points for faster training and inference.
    """
    def __init__(
            self,
            model_config: ModelConfig = ModelConfig(),
            num_inducing: int = 200,
    ):
        """ """
        super(MarginalGPyTorch, self).__init__(model_config=model_config)
        self.build_datamanager(model_config)
        self.num_inducing = num_inducing
        

    def build_model(self, X, y, y_unc=None) -> gpytorch.models.ApproximateGP:
        """Build variational sparse GP version of RatingGP
        """
        # Select inducing points using k-means++ initialization
        inducing_points = self._select_inducing_points(X, self.num_inducing)
        
        # Create variational distribution and strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=self.num_inducing
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            base_variational_strategy=gpytorch.variational.IndependentMultitaskVariationalStrategy(
                base_variational_strategy=gpytorch.variational.VariationalStrategy(
                    model=None,  # Will be set by the model
                    inducing_points=inducing_points,
                    variational_distribution=variational_distribution,
                    learn_inducing_locations=True
                ),
                num_tasks=1
            ) if y.dim() > 1 else gpytorch.variational.VariationalStrategy(
                model=None,  # Will be set by the model
                inducing_points=inducing_points,
                variational_distribution=variational_distribution,
                learn_inducing_locations=True
            ),
            num_data=y.size(0)
        )

        # Use Gaussian likelihood (no fixed noise for variational)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=gpytorch.priors.HalfNormalPrior(scale=0.1),
        )

        model = VariationalGPModel(inducing_points, variational_strategy)
        return model

    def _select_inducing_points(self, X, num_inducing):
        """Select inducing points using k-means++ initialization"""
        from sklearn.cluster import KMeans
        
        # Convert to numpy for sklearn
        X_np = X.detach().cpu().numpy()
        
        # Use k-means to select inducing points
        kmeans = KMeans(n_clusters=num_inducing, init='k-means++', random_state=42)
        kmeans.fit(X_np)
        
        # Convert back to tensor
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=X.dtype, device=X.device)
        return inducing_points

    def fit(self, covariates, target, target_unc=None, iterations=1000, 
            optimizer='adam', learning_rate=0.01, early_stopping=True, patience=50):
        """Fit the variational GP model with ELBO optimization"""
        
        # Prepare data
        self.fit_dm(covariates, target, target_unc)
        train_x = torch.tensor(self.X, dtype=torch.float32)
        train_y = torch.tensor(self.y, dtype=torch.float32)
        
        # Build model
        self.model = self.build_model(train_x, train_y, target_unc)
        self.model.train()
        self.likelihood.train()

        # Use variational ELBO as the loss
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        # Set up optimizer
        if optimizer == 'adam':
            opt = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=learning_rate)
        elif optimizer == 'adamw':
            opt = torch.optim.AdamW([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(iterations):
            opt.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            opt.step()
            
            if i % 100 == 0:
                print(f'Iter {i:4d}/{iterations} - Loss: {loss.item():.3f}')
            
            # Early stopping
            if early_stopping:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        print(f'Early stopping at iteration {i}')
                        break

        self.model.eval()
        self.likelihood.eval()


class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, variational_strategy):
        super(VariationalGPModel, self).__init__(variational_strategy)
        
        # Store inducing points info
        self.inducing_points = inducing_points
        n_d = inducing_points.shape[1]  # number of dimensions
        assert n_d == 2, "Only two dimensions supported"

        self.dims = np.arange(n_d)
        self.time_dim = [self.dims[0]]
        self.stage_dim = [self.dims[1]]

        self.powerlaw = PowerLawTransform()
        self.mean_module = NoOpMean()

        # Use stage for sigmoid kernel constraint
        stage = inducing_points[:, self.stage_dim[0]]
        b_min = torch.quantile(stage, 0.10)
        b_max = torch.quantile(stage, 0.90) 
        b_bottom = torch.quantile(stage, 0.25)  # 25th percentile

        # Create sigmoid kernel for gating (shared switchpoint) with prior toward bottom
        b_constraint = gpytorch.constraints.Interval(-10.0, 10.0)
        sigmoid_lower = SigmoidKernel(
            active_dims=self.stage_dim,
            b_constraint=b_constraint,
            b_prior=NormalPrior(loc=float(b_bottom), scale=float((b_max - b_min) / 8)),
        )
        # Initialize b parameter randomly between 1.2 and 1.8 using initialize
        init_val = torch.rand(1) * 0.6 + 1.2
        try:
            sigmoid_lower.initialize(raw_b=sigmoid_lower.raw_b_constraint.inverse_transform(init_val))
        except Exception:
            # Best-effort fallback
            with torch.no_grad():
                sigmoid_lower.raw_b = sigmoid_lower.raw_b_constraint.inverse_transform(init_val)

        sigmoid_upper = InvertedSigmoidKernel(
            sigmoid_kernel=sigmoid_lower,
            active_dims=self.stage_dim,
        )

        # Compose the upper and lower kernel branches with PowerLawWarp
        upper_kernel = (
            self.cov_base(eta_prior=HalfNormalPrior(scale=1.0))
            + self.cov_periodic(eta_prior=HalfNormalPrior(scale=0.2))
            + self.cov_bend(eta_prior=HalfNormalPrior(scale=0.2))
        )
        upper_kernel_warped = PowerLawWarpKernel(upper_kernel, self.powerlaw, self.stage_dim[0])

        lower_kernel = self.cov_shift(
            eta_prior=HalfNormalPrior(scale=1.0),
            time_prior=GammaPrior(concentration=1, rate=10),
        )
        lower_kernel_warped = PowerLawWarpKernel(lower_kernel, self.powerlaw, self.stage_dim[0])

        self.covar_module = (
            sigmoid_lower * lower_kernel_warped
            + sigmoid_upper * upper_kernel_warped
        )

    def forward(self, x):
        # Avoid in-place clamp of Parameters; prefer constrained param or
        # compute a clamped view when needed.
        try:
            b_val = self.powerlaw.b_constraint.transform(self.powerlaw.b)
            # optionally bound b_val to [1.2, 2.5] for safety in the forward
            b_val = torch.clamp(b_val, min=1.2, max=2.5)
            self.powerlaw_b_view = b_val
        except Exception:
            self.powerlaw_b_view = torch.clamp(self.powerlaw.b, min=1.2, max=2.5)
        x_t = x.clone()
        # powerlaw forward uses its own constrained b when available
        x_t[:, self.stage_dim] = self.powerlaw(x_t[:, self.stage_dim])
        q = x_t[:, self.stage_dim]
        mean_x = self.mean_module(q)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def cov_shift(self, eta_prior=None, time_prior=None):
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=0.3) 
        if time_prior is None:
            time_prior = GammaPrior(concentration=1, rate=7)

        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=GammaPrior(concentration=1, rate=1),
            ) *
            MaternKernel(
                active_dims=self.time_dim,
                lengthscale_prior=time_prior,
                nu=1.5,
            ),
            outputscale_prior=eta_prior,
        )
 
    def cov_bend(self, eta_prior=None):
        """Smooth, time-dependent bending kernel for switchpoint."""
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=0.2) 

        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=GammaPrior(concentration=3, rate=1),
            ) *
            MaternKernel(
                active_dims=self.time_dim,
                lengthscale_prior=GammaPrior(concentration=3, rate=2),
            ),
            outputscale_prior=eta_prior,
        )
    
    def cov_periodic(self, ls_prior=None, eta_prior=None):
        """Smooth, time-dependent periodic kernel for seasonal effects."""
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=0.5) 
        if ls_prior is None:
            ls_prior = GammaPrior(concentration=2, rate=4)

        return ScaleKernel(
            PeriodicKernel(
                active_dims=self.time_dim,
                period_length_prior=NormalPrior(loc=1.0, scale=0.1),  # ~1 year
                lengthscale_prior=ls_prior,
            ),
            outputscale_prior=eta_prior,
        )
    
    def cov_base(self, eta_prior=None):
        """Smooth, time-independent base rating curve using a Matern kernel on stage."""
        if eta_prior is None:
            eta = HalfNormalPrior(scale=1.0)
        else:
            eta = eta_prior

        ls = GammaPrior(concentration=5, rate=1)
        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )
