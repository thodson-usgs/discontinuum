import gpytorch
import numpy as np
import torch
from discontinuum.engines.gpytorch import MarginalGPyTorch
from gpytorch.kernels import (
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.priors import (
    GammaPrior,
    HalfNormalPrior,
    NormalPrior,
)

from loadest_gp.models.base import LoadestDataMixin, ModelConfig
from loadest_gp.plot import LoadestPlotMixin


class LoadestGPMarginalGPyTorch(
    LoadestDataMixin,
    LoadestPlotMixin,
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

    def build_model(self, X, y) -> gpytorch.models.ExactGP:
        """Build marginal likelihood version of LoadestGP
        """
        # noise_prior = HalfNormalPrior(scale=0.01)
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
        #     noise_prior=noise_prior,
        # )

        noise = 0.1**2 * torch.ones(y.shape[0]).reshape(1, -1)
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
        self.dims = np.arange(n_d)
        self.time_dim = [self.dims[0]]
        self.cov_dims = self.dims[1:]

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = (
            # may omit trend: wall 1.08s vs 990ms without,
            # but much faster to compile
            #self.cov_trend()
            self.cov_seasonal()
            + self.cov_covariates()
            + self.cov_residual()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def cov_trend(self):
        eta = HalfNormalPrior(scale=1)
        ls = GammaPrior(concentration=4, rate=1)

        return ScaleKernel(
            RBFKernel(
                # active_dims=(0),
                active_dims=self.time_dim,
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )

    def cov_seasonal(self):
        # TODO add lengthscale priors
        eta = HalfNormalPrior(scale=1)
        period = NormalPrior(loc=1, scale=0.01)

        return ScaleKernel(
            PeriodicKernel(
                period_length_prior=period,
                active_dims=self.time_dim,
                )
            * MaternKernel(
                nu=2.5,
                active_dims=self.time_dim
               ),
           outputscale_prior=eta,
        )

    def cov_covariates(self):
        eta = HalfNormalPrior(scale=2)
        ls = GammaPrior(concentration=2, rate=3)  # alpha, beta

        return ScaleKernel(
            RBFKernel(
                ard_num_dims=self.cov_dims.shape[0],
                lengthscale_prior=ls,
                active_dims=self.cov_dims,
            ),
            outputscale_prior=eta,
        )

    def cov_residual(self):
        eta = HalfNormalPrior(scale=0.2)
        ls = GammaPrior(concentration=2, rate=10)

        return ScaleKernel(
            MaternKernel(
                ard_num_dims=self.dims.shape[0],
                nu=1.5,
                active_dims=self.dims,
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )
