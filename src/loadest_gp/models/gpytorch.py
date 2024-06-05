import gpytorch
import numpy as np
import torch

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

from discontinuum.engines.gpytorch import MarginalGPyTorch

from loadest_gp.plot import LoadestPlotMixin
from loadest_gp.models.base import LoadestDataMixin


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
    def __init__(self, model_config=None):
        """ """
        super(MarginalGPyTorch, self).__init__(model_config=model_config)
        self.build_datamanager()

    def build_model(self, X, y) -> gpytorch.models.ExactGP:
        """Build marginal likelihood version of LoadestGP
        """
        #train_x = torch.tensor(X, dtype=torch.float32)
        #train_y = torch.tensor(y, dtype=torch.float32)

        noise_prior = HalfNormalPrior(scale=0.01)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior
        )

        #model = ExactGPModel(train_x, train_y, self.likelihood)
        model = ExactGPModel(X, y, self.likelihood)

        return model


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        n_d = train_x.shape[1]  # number of dimensions
        dims = np.arange(n_d)
        time_dim = [dims[0]]
        cov_dims = dims[1:]

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = (
            self.cov_trend()
            + self.cov_seasonal()
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
                active_dims=(0),
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )

    def cov_seasonal(self):
        # TODO add priors
        period = NormalPrior(loc=1, scale=0.01)

        return ScaleKernel(
            PeriodicKernel(period_length_prior=period, active_dims=(0))
            * MaternKernel(nu=2.5, active_dims=(0)),
            # outputscale_prior=eta
        )

    def cov_covariates(self):
        eta = HalfNormalPrior(scale=2)
        ls = GammaPrior(concentration=2, rate=3)  # alpha, beta

        return ScaleKernel(
            RBFKernel(
                lengthscale_prior=ls,
                active_dims=(1),
            ),
            outputscale_prior=eta,
        )

    def cov_residual(self):
        eta = HalfNormalPrior(scale=0.2)
        ls = GammaPrior(concentration=2, rate=10)

        return ScaleKernel(
            MaternKernel(
                ard_num_dims=2,
                nu=1.5,
                active_dims=(0, 1),
                lengthscale_prior=ls,
            ),
            outputscale_prior=eta,
        )
