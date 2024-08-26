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

from rating_gp.models.base import RatingDataMixin, ModelConfig
from rating_gp.plot import RatingPlotMixin
from rating_gp.models.kernels import StageTimeKernel, SigmoidKernel

import torch.nn.functional as F

class PowerLawTransform(torch.nn.Module):
    """
    """
    def __init__(self):
        super(PowerLawTransform, self).__init__()
        self.a = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))
        self.c = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # self.c.data = torch.clamp(self.c.data, max=x.min()-1e-6)
        # return self.a + (self.b * torch.log(x - self.c))
        m = x > self.c  # mask flow state: stage > c
        output = torch.empty_like(x)
        # flow state
        output[m] = self.a + (self.b * torch.log(x[m] - self.c))
        # no-flow state
        zero_flow_value = 1e-6  # TODO set in config and provider
        output[~m] = np.log(zero_flow_value)  # avoid log(0) error
        return output


class PowerLawLayer(PowerLawTransform):
     def forward(self, c, x):
        """
        Parameters
        ----------
        a : torch.Tensor
            Parameters for the power law transformation
        x : torch.Tensor
            2D time and stage data
        """
        self.b.data = torch.clamp(self.b.data, min=1.5, max=2.5)  #TODO verify LeCoz 2014 range
        h_max = x[:, 1].max()
        c = self.c + c.flatten()  # add flatten
        # TODO clamp is set too high such that the model is not utilizing the
        # power law but rather behaves like a constant mean model, because the
        # model too easily adopts the no-flow state, even when during high
        # flows. Might try setting clamp to the parameter of the sigmoid.
        # TODO An alternative idea, is that we just need the model to follow the data
        # at low stages and power-law theory at high stages. This might be achieved
        # without a neural network by switching the mean function at an
        # intermediate stage, like that of the sigmoid kernel. At low stages,
        # apply a constant mean, and at high stages apply the power law.
        # TODO Another idea, and probably the simplest, but least-physical, is
        # to add an arbitrary offset in the stage transformation, thereby giving
        # the model a little more tolerance by keeping c below the data range.
        c.data = torch.clamp(c.data, min=0, max=h_max)
        m = x[:, 1] > c  # mask the flow state
        x_t = x[m]

        output = torch.empty_like(x[:, 1])
        # flow state
        output[m] = self.a
        output[m] += self.b * torch.log(x_t[:, 1] - c[m])
        # no-flow state
        zero_flow_value = 1e-6  # TODO set in config
        output[~m] = np.log(zero_flow_value)  # avoid log(0) error
        return output


class NeuralPowerLaw(torch.nn.Module):
    def __init__(self):
        super(NeuralPowerLaw, self).__init__()
        # 1st test was 500>50>3; ~150 it/s
        # 2nd was 1000>500>50>3; ~110 it/s
        self.l0 = torch.nn.Linear(1, 500)
        self.l1 = torch.nn.Linear(500, 50)
        self.l2 = torch.nn.Linear(50, 1)
        self.p0 = PowerLawLayer()

    def forward(self, x):
        t = x[:, 0].view(-1, 1)  # time
        a = self.l0(t)
        a = F.relu(a)
        a = self.l1(a)
        a = F.relu(a)
        a = self.l2(a)
        return self.p0(a, x)


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

        #self.powerlaw = PowerLawTransform()
        self.powerlaw = NeuralPowerLaw()

        # self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean(input_size=1)
        self.mean_module = NoOpMean()

        # self.covar_module = (
        #     (self.cov_stage() * self.cov_stagetime())
        #     + self.cov_residual()
        # )
      
        # Stage * time kernel with large time length
        # + stage * time kernel only at low stage with smaller time length
        self.covar_module = (
            (self.cov_stage()
             * self.cov_time(ls_prior=GammaPrior(concentration=10, rate=5)))
            + (self.cov_stage()
               * self.cov_time(ls_prior=GammaPrior(concentration=2, rate=5))
               * SigmoidKernel(
                   active_dims=self.stage_dim,
                   # a_prior=NormalPrior(loc=20, scale=1),
                   b_constraint=gpytorch.constraints.Interval(
                       train_x[:, self.stage_dim].min(),
                       train_x[:, self.stage_dim].max()
                   ),
               )
              )
        )

    def forward(self, x):
        #x_t = x.clone()
        #x_t[:, self.stage_dim] = self.powerlaw(x_t[:, self.stage_dim])
        #mean_x = self.mean_module(x_t[:, self.stage_dim])
        #covar_x = self.covar_module(x_t)
        x_t = self.powerlaw(x)
        mean_x = self.mean_module(x_t)
        # TODO: also try passing x_t to covar but with correct dims
        covar_x = self.covar_module(x)
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
