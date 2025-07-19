import gpytorch
import numpy as np
import torch
from discontinuum.engines.gpytorch import MarginalGPyTorch, NoOpMean


from gpytorch.kernels import (
    MaternKernel,
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
from rating_gp.models.kernels import SigmoidKernel


class PowerLawTransform(torch.nn.Module):
    """
    """
    def __init__(self):
        super(PowerLawTransform, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))
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
        super(MarginalGPyTorch, self).__init__(model_config=model_config)
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
            noise_prior=gpytorch.priors.HalfNormalPrior(scale=0.005),
        )

        model = ExactGPModel(X, y, self.likelihood)

        return model


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
        sigmoid_shift = SigmoidKernel(
            active_dims=self.stage_dim,
            b_constraint=gpytorch.constraints.Interval(b_min, b_max),
        )

        # Inverted sigmoid kernel as a proper GPyTorch kernel
        class InvertedSigmoidKernel(SigmoidKernel):
            def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
                # Use the same b parameter as sigmoid_shift
                self.raw_b = sigmoid_shift.raw_b
                # Standard sigmoid
                x1_ = 1/(1 + torch.exp(self.a * (x1 - self.b)))
                x2_ = 1/(1 + torch.exp(self.a * (x2 - self.b)))
                # Invert: use 1 - sigmoid
                x1_inv = 1.0 - x1_
                x2_inv = 1.0 - x2_
                if diag:
                    return (x1_inv * x2_inv).squeeze(-1)
                else:
                    return torch.matmul(x1_inv, x2_inv.transpose(-2, -1))

        sigmoid_base_inverted = InvertedSigmoidKernel(
            active_dims=self.stage_dim,
            b_constraint=gpytorch.constraints.Interval(b_min, b_max),
        )
        
        self.covar_module = (
            # core time kernel
            (
                 self.cov_time(
                     #ls_prior=GammaPrior(concentration=2, rate=1),
                     ls_prior=GammaPrior(concentration=3, rate=1),
                     eta_prior=HalfNormalPrior(scale=0.3),
                 ) 
                 *
                 self.cov_stage(ls_prior=GammaPrior(concentration=3, rate=2))
                 #self.cov_stage(ls_prior=GammaPrior(concentration=2, rate=1))
             ) * sigmoid_base_inverted
             # shift component gated by sigmoid (active below switchpoint)
             + (
                self.cov_time(
                    ls_prior=GammaPrior(concentration=2, rate=5),
                    eta_prior=HalfNormalPrior(scale=1),
                )
            ) * sigmoid_shift
            # base curve component gated by inverted sigmoid (active above switchpoint)
            + (
                (self.cov_base() + self.cov_periodic())
            ) * sigmoid_base_inverted
            # additive periodic component for seasonal effects
        )


    def forward(self, x):
        self.powerlaw.b.data.clamp_(1.2, 2.5)
        #x = x.clone()
        #q = self.powerlaw(x[:, self.stage_dim])
        #x_t[:, self.stage_dim] = self.warp_stage_dim(x_t[:, self.stage_dim])
        x_t = x.clone()
        x_t[:, self.stage_dim] = self.powerlaw(x_t[:, self.stage_dim])
        q = x_t[:, self.stage_dim]
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
                nu=2.5,  # Smoother kernel (was nu=1.5)
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
                nu=1.5, # was 1.5 XXX
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
            ls_prior = GammaPrior(concentration=3, rate=1)

        return ScaleKernel(
            PeriodicKernel(
                active_dims=self.time_dim,
                period_length_prior=NormalPrior(loc=1.0, scale=0.1),  # ~1 year
                # lengthscale_prior=GammaPrior(concentration=2, rate=4),
            ),
            # *
            # MaternKernel(
            #     active_dims=self.stage_dim,
            #     lengthscale_prior=ls_prior,
            #     nu=2.5,  # Smoother kernel (was nu=1.5)
            # ),
            outputscale_prior=HalfNormalPrior(scale=0.5),
        )
    
    def cov_base(self):
        """
        Smooth, time-independent base rating curve using a Matern kernel on stage.
        """
        # Base should capture most variation
        eta = HalfNormalPrior(scale=1)
        ls = GammaPrior(concentration=10, rate=1)
        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )
 