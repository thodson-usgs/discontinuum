import torch
import gpytorch
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import Noise
from gpytorch.constraints import GreaterThan
from linear_operator.operators import DiagLinearOperator


class LearnedHeteroskedasticNoise(Noise):
    """Learnable per-observation noise model.

    Stores a noise parameter for each training observation and optimizes it
    during GP training. The noise values are constrained to be positive to
    ensure numerical stability.
    """

    def __init__(self, noise: torch.Tensor, noise_prior=None, noise_constraint=None):
        super().__init__()
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)
        noise = noise.reshape(-1)
        self.register_parameter(
            name="raw_noise",
            parameter=torch.nn.Parameter(torch.zeros_like(noise))
        )
        self.register_constraint("raw_noise", noise_constraint)
        if noise_prior is not None:
            self.register_prior(
                "noise_prior", noise_prior, self._noise_param, self._noise_closure
            )
        self._set_noise(noise)

    def _noise_param(self, m):
        return m.noise

    def _noise_closure(self, m, v):
        return m._set_noise(v)

    @property
    def noise(self):
        n = self.raw_noise_constraint.transform(self.raw_noise)
        return torch.nan_to_num(n, nan=1e-4)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))
        return self

    def forward(self, *params, shape=None, noise=None, **kwargs):
        if noise is not None:
            return DiagLinearOperator(noise.reshape(-1))
        if shape is not None and shape[-1] != self.noise.numel():
            default = self.noise.mean().repeat(shape[-1])
            return DiagLinearOperator(default)
        return DiagLinearOperator(self.noise)


class HeteroskedasticGaussianLikelihood(_GaussianLikelihoodBase):
    """Gaussian likelihood with per-observation learnable noise."""

    def __init__(self, noise: torch.Tensor, noise_prior=None, noise_constraint=None):
        noise_covar = LearnedHeteroskedasticNoise(
            noise=noise,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self):
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar._set_noise(value)
