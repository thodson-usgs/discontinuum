"""Custom prior distributions for rating_gp models."""

import math

import torch
from torch.distributions import Distribution, constraints
from torch.nn import Module as TModule

from gpytorch.priors import Prior
from gpytorch.priors.utils import _bufferize_attributes


class _Horseshoe(Distribution):
    """Half-horseshoe distribution.

    This distribution has support on the positive reals and provides strong
    shrinkage around zero with very heavy tails, making it useful as a prior for
    scale parameters that may occasionally take on extremely large values.

    The probability density function is

    .. math::

        f(x \mid s) = \frac{2}{\pi s} \log\left(1 + \frac{s^2}{x^2}\right),

    where ``s`` is a positive scale parameter and ``x > 0``.
    """

    arg_constraints = {"scale": constraints.positive}
    support = constraints.positive

    def __init__(self, scale, validate_args=False):
        self.scale = torch.as_tensor(scale)
        batch_shape = self.scale.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):  # pragma: no cover - sampling is stochastic
        shape = sample_shape + self.batch_shape
        lam = torch.distributions.HalfCauchy(torch.ones_like(self.scale)).rsample(shape)
        tau = torch.distributions.HalfCauchy(self.scale).rsample(shape)
        z = torch.abs(torch.randn(shape, dtype=self.scale.dtype, device=self.scale.device))
        return tau * lam * z

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        scale = self.scale
        return (
            math.log(2.0)
            - math.log(math.pi)
            - torch.log(scale)
            + torch.log(torch.log1p((scale / value) ** 2))
        )


class HorseshoePrior(Prior, _Horseshoe):
    """Horseshoe prior for non-negative parameters."""

    def __init__(self, scale=1.0, validate_args=None, transform=None):
        TModule.__init__(self)
        _Horseshoe.__init__(self, scale=scale, validate_args=validate_args)
        _bufferize_attributes(self, ("scale",))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return HorseshoePrior(self.scale.expand(batch_shape))


__all__ = ["HorseshoePrior"]

