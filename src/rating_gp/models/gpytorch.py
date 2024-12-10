import gpytorch
import numpy as np
import torch
from discontinuum.engines.gpytorch import MarginalGPyTorch, NoOpMean

from gpytorch.kernels import (
    MaternKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.priors import (
    GammaPrior,
    HalfNormalPrior,
    NormalPrior,
)

from linear_operator.operators import MatmulLinearOperator
from rating_gp.models.base import RatingDataMixin, ModelConfig
from rating_gp.plot import RatingPlotMixin
from rating_gp.models.kernels import StageTimeKernel, SigmoidKernel, LogWarp, TanhWarp


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
            learn_additional_noise=False,
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

        # self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean(input_size=1)
        self.mean_module = NoOpMean()

        #self.warp_stage_dim = TanhWarp()
        #self.warp_stage_dim = LogWarp()

        # self.covar_module = (
        #     (self.cov_stage() * self.cov_stagetime())
        #     + self.cov_residual()
        # )
         
        # Stage * time kernel with large time length
        # + stage * time kernel only at low stage with smaller time length.
        # Note that stage gets transformed to q, so the kernel is actually
        # q * time
        self.covar_module = (
            (self.cov_stage()
             * self.cov_time(ls_prior=GammaPrior(concentration=10, rate=5)))
            + (self.cov_stage()
               * self.cov_time(ls_prior=GammaPrior(concentration=2, rate=5))
               * SigmoidKernel(
                   active_dims=self.stage_dim,
                   # a_prior=NormalPrior(loc=20, scale=1),
                   b_constraint=gpytorch.constraints.Interval(
                       train_y.min(),
                       train_y.max(),
                   ),
               )
              )
        )


    def forward(self, x):
        self.powerlaw.b.data.clamp_(1.5, 2.5)
        #x = x.clone()
        #q = self.powerlaw(x[:, self.stage_dim])
        #x_t[:, self.stage_dim] = self.warp_stage_dim(x_t[:, self.stage_dim])
        x_t = x.clone()
        x_t[:, self.stage_dim] = self.powerlaw(x_t[:, self.stage_dim])
        q = x_t[:, self.stage_dim]

        mean_x = self.mean_module(q)
        covar_x = self.covar_module(x_t)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def cov_stage(self, ls_prior=None):
        eta = HalfNormalPrior(scale=1)

        return ScaleKernel(
            MaternKernel(
                active_dims=self.stage_dim,
                lengthscale_prior=ls_prior,
            ),
            outputscale_prior=eta,
        )

    def cov_time(self, ls_prior=None):
        eta = HalfNormalPrior(scale=1)

        return ScaleKernel(
            MaternKernel(
                active_dims=self.time_dim,
                lengthscale_prior=ls_prior,
            ),
            outputscale_prior=eta,
        )

    def cov_stagetime(self):
        eta = HalfNormalPrior(scale=1)
        ls = GammaPrior(concentration=2, rate=1)

        return ScaleKernel(
            StageTimeKernel(
                active_dims=self.dims,
                # lengthscale_prior=ls,
            ),
            # outputscale_prior=eta,
        )

    def cov_residual(self):
        eta = HalfNormalPrior(scale=0.2)
        ls = GammaPrior(concentration=2, rate=10)

        return ScaleKernel(
            MaternKernel(
                ard_num_dims=2,
                nu=1.5,
                active_dims=self.dims,
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )
