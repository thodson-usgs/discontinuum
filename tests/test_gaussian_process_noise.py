import torch
import gpytorch
import pytest

from rating_gp.models.gpytorch import ExactGPModel
from rating_gp.models.noise import GaussianProcessGaussianLikelihood


def test_gaussian_process_gaussian_likelihood_requires_inputs():
    train_x = torch.rand(5, 2)
    train_y = torch.randn(5)
    likelihood = GaussianProcessGaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    output = model(train_x)

    with pytest.raises(TypeError):
        mll(output, train_y)

    # Should succeed when inputs are provided
    mll(output, train_y, train_x)


def test_gaussian_process_gaussian_likelihood_has_priors():
    likelihood = GaussianProcessGaussianLikelihood()
    covar = likelihood.covar_module
    base = covar.base_kernel
    assert isinstance(base.lengthscale_prior, gpytorch.priors.SmoothedBoxPrior)
    assert isinstance(covar.outputscale_prior, gpytorch.priors.SmoothedBoxPrior)
    # priors should keep lengthscale and outputscale within reasonable bounds
    assert float(base.lengthscale_prior.a) > 0.0
    assert float(base.lengthscale_prior.b) <= 0.5
    assert float(covar.outputscale_prior.a) > 0.0
    assert float(covar.outputscale_prior.b) <= 0.25
