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
            noise_prior=gpytorch.priors.HalfNormalPrior(scale=0.03),
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
        sigmoid_lower = SigmoidKernel(
            active_dims=self.stage_dim,
            b_constraint=gpytorch.constraints.Interval(b_min, b_max),
        )
 
        sigmoid_upper = InvertedSigmoidKernel(
            sigmoid_kernel=sigmoid_lower,
            active_dims=self.stage_dim,
            b_constraint=gpytorch.constraints.Interval(b_min, b_max),
        )
 
        self.covar_module = (
            sigmoid_lower * (
                self.cov_shift(
                    #eta_prior=HalfNormalPrior(scale=2.0),
                    eta_prior=HalfNormalPrior(scale=1.0),
                    time_prior=GammaPrior(concentration=1, rate=7),
                )
            )
            + sigmoid_upper * (
                self.cov_base(eta_prior=HalfNormalPrior(scale=4.0))
                +
                self.cov_periodic(eta_prior=HalfNormalPrior(scale=0.2))
                +
                self.cov_bend(eta_prior=HalfNormalPrior(scale=0.2))
            )
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
    
    def cov_shift(self, eta_prior=None, time_prior=None):
        if eta_prior is None:
            eta_prior = HalfNormalPrior(scale=0.3) 

        if time_prior is None:
            time_prior = GammaPrior(concentration=1, rate=7)

        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=GammaPrior(concentration=1, rate=1),
                #nu=1.5,
            ) *
            MaternKernel(
                active_dims=self.time_dim,
                # extreme prior for fast shift at 12413470
                #lengthscale_prior=GammaPrior(concentration=0.1, rate=100),
                lengthscale_prior=time_prior,
                nu=1.5,
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
                lengthscale_prior=GammaPrior(concentration=3, rate=1),
            ) *
            MaternKernel(
                active_dims=self.time_dim,
                lengthscale_prior=GammaPrior(concentration=3, rate=2),
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
            ls_prior = GammaPrior(concentration=2, rate=4)

        return ScaleKernel(
            PeriodicKernel(
                active_dims=self.time_dim,
                period_length_prior=NormalPrior(loc=1.0, scale=0.1),  # ~1 year
                lengthscale_prior=ls_prior,
            ),
            # *
            # MaternKernel(
            #     active_dims=self.stage_dim,
            #     lengthscale_prior=ls_prior,
            #     nu=2.5,  # Smoother kernel (was nu=1.5)
            # ),
            outputscale_prior=eta_prior,
        )
    
    def cov_base(self, eta_prior=None):
        """
        Smooth, time-independent base rating curve using a Matern kernel on stage.
        """
        # Base should capture most variation
        if eta_prior is None:
            eta = HalfNormalPrior(scale=1.0)
        else:
            eta = eta_prior

        ls = GammaPrior(concentration=5, rate=1) # amazing!
        # ls = GammaPrior(concentration=6, rate=1)
        return ScaleKernel(
            #RBFKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )
