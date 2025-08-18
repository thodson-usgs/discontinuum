import gpytorch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_operator.operators import MatmulLinearOperator, to_dense

from gpytorch.constraints import Positive
from gpytorch.priors import GammaPrior

try:
    from linear_operator.operators import DiagLinearOperator, ZeroLinearOperator
except ImportError:
    from gpytorch.lazy import DiagLazyTensor as DiagLinearOperator
    from gpytorch.lazy import ZeroLazyTensor as ZeroLinearOperator


class TanhWarp(torch.nn.Module):

    def __init__(self):
        super(TanhWarp, self).__init__()
        self.a = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))
        self.c = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        return x + (self.a * torch.tanh(self.b * (x - self.c)))


class LogWarp(torch.nn.Module):
    """Logarithmic Warp

    Note: good smoother
    """
    def __init__(self):
        super(LogWarp, self).__init__()
        self.a = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.a * torch.log(x)


class StageTimeKernel(gpytorch.kernels.Kernel):
    """A time RBF kernel with a stage-variable length scale.

    A scalar length scale is multiplied by the stage, which is taken to a
    variable power, to impose a stage variablity.
    """
    has_lengthscale = True

    def __init__(self,
                 a_prior=None,
                 a_constraint=gpytorch.constraints.Positive(),
                 **kwargs):
        """Initialize the kernel

        Parameters
        ----------
        a_prior : Prior
            The prior to impose on the power variable.
        a_constraint : Constraint
            The constraint to impose on the power variable
        """
        super().__init__(**kwargs)

        # register the raw parameters
        self.register_parameter(
            name='raw_a',
            parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1))
        )

        # register the constraints
        # if a_constraint is not None:
        self.register_constraint("raw_a", a_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if a_prior is not None:
            self.register_prior(
                "a_prior",
                a_prior,
                lambda m: m.a,
                lambda m, v : m._set_a(v),
            )

    # now set up the 'actual' paramter
    @property
    def a(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_a_constraint.transform(self.raw_a)

    @a.setter
    def a(self, value):
        return self._set_a(value)

    def _set_a(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_a=self.raw_a_constraint.inverse_transform(value))

    # create a RBF Kernel that has a stage dependent scale length
    def forward(self, x1, x2, **params):
        dims = np.arange(x1.shape[1])
        time_dim = [dims[0]]
        stage_dim = [dims[1]]

        # make the scale length for the time dimension dependent on stage
        # shift stage to have a minimum value of 1
        x1_stage = x1[:, stage_dim] - x1[:, stage_dim].min() + 1
        x2_stage = x2[:, stage_dim] - x2[:, stage_dim].min() + 1
        
        x1_stage_dep_time_lengthscale = self.lengthscale * x1_stage ** self.a
        x2_stage_dep_time_lengthscale = self.lengthscale * x2_stage ** self.a

        # apply lengthscale
        x1_ = x1[:, time_dim].div(x1_stage_dep_time_lengthscale)
        x2_ = x2[:, time_dim].div(x2_stage_dep_time_lengthscale)

        # calculate the distance between inputs
        diff = self.covar_dist(x1_, x2_, square_dist=True, **params)
        return diff.div_(-2).exp_()


class PowerLawKernel(gpytorch.kernels.Kernel):
    """Power Law Kernel

    This kernel is the rating curve power law equivalent to the linear kernel.
    The power law equation is given by:

        f(x) = a + (b * ln(x - c))
    """
    def __init__(self,
                 a_prior=None,
                 a_constraint=None,
                 b_prior=None,
                 b_constraint=gpytorch.constraints.Positive(),
                 c_prior=None,
                 c_constraint=gpytorch.constraints.Positive(),
                 **kwargs):
        """Initialize the kernel

        Parameters
        ----------
        a_prior : Prior
            The prior to impose on `a`.
        a_constraint : Constraint
            The constraint to impose on `a`
        b_prior : Prior
            The prior to impose on `b`.
        b_constraint : Constraint
            The constraint to impose on `b`
        c_prior : Prior
            The prior to impose on `c`.
        c_constraint : Constraint
            The constraint to impose on `c`
        """
        super().__init__(**kwargs)

        # register the raw parameters
        self.register_parameter(
            name='raw_a',
            parameter=torch.nn.Parameter(torch.rand(*self.batch_shape, 1, 1))
        )
        b_start = torch.rand(*self.batch_shape, 1, 1)
        self.register_parameter(
            name='raw_b',
            parameter=torch.nn.Parameter(b_start)
        )
        self.register_parameter(
            name='raw_c',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # register the constraints
        if a_constraint is None:
            a_constraint = gpytorch.constraints.Interval(-10, 10)
        self.register_constraint("raw_a", a_constraint)
        self.register_constraint("raw_b", b_constraint)
        self.register_constraint("raw_c", c_constraint)

        # set the parameter prior
        if a_prior is not None:
            self.register_prior(
                "a_prior",
                a_prior,
                lambda m: m.a,
                lambda m, v : m._set_a(v),
            )
        if b_prior is not None:
            self.register_prior(
                "b_prior",
                b_prior,
                lambda m: m.b,
                lambda m, v : m._set_b(v),
            )
        if c_prior is not None:
            self.register_prior(
                "c_prior",
                c_prior,
                lambda m: m.c,
                lambda m, v : m._set_c(v),
            )

    # set the actual parameters
    @property
    def a(self):
        return self.raw_a_constraint.transform(self.raw_a)

    @a.setter
    def a(self, value):
        return self._set_a(value)

    def _set_a(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a)
        self.initialize(raw_a=self.raw_a_constraint.inverse_transform(value))

    @property
    def b(self):
        return self.raw_b_constraint.transform(self.raw_b)

    @b.setter
    def b(self, value):
        return self._set_b(value)

    def _set_b(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_b)
        self.initialize(raw_b=self.raw_b_constraint.inverse_transform(value))

    @property
    def c(self):
        return self.raw_c_constraint.transform(self.raw_c)

    @c.setter
    def c(self, value):
        return self._set_c(value)

    def _set_c(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_c)
        self.initialize(raw_c=self.raw_c_constraint.inverse_transform(value))

    # the kernel function
    def forward(self, x1, x2, diag=False, **params):
        x1_ = self.a + (self.b * torch.log(x1 - self.c))
        x2_ = self.a + (self.b * torch.log(x2 - self.c))

        # Calculate cross product
        prod = MatmulLinearOperator(x1_, x2_.transpose(-2, -1))
        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod


class SigmoidKernel(gpytorch.kernels.Kernel):
    """Sigmoid gating kernel for switchpoints.

    K(x1, x2) = s(x1) s(x2)^T, where s(x) = 1 / (1 + exp(sharpness * (x - b))).

    The "sharpness" is fixed (non-trainable) for stability; the location parameter b
    is constrained and can carry an optional prior.
    """

    def __init__(
        self,
        b_constraint,
        b_prior=None,
        sharpness: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Fixed sharpness as a buffer for device/dtype consistency
        self.register_buffer("sharpness", torch.tensor(float(sharpness)))

        # register raw_b parameter with an initial value within [l, u]
        u = b_constraint.upper_bound
        l = b_constraint.lower_bound
        init_b = l + torch.rand((*self.batch_shape, 1, 1)) * (u - l)
        self.register_parameter(
            name="raw_b",
            parameter=torch.nn.Parameter(torch.zeros((*self.batch_shape, 1, 1))),
        )

        if b_constraint is not None:
            self.register_constraint("raw_b", b_constraint)
        self.initialize(raw_b=self.raw_b_constraint.inverse_transform(init_b))

        # Attach prior if provided
        if b_prior is not None:
            self.register_prior(
                "b_prior",
                b_prior,
                lambda m: m.b,
                lambda m, v: m._set_b(v),
            )

    # set the 'actual' paramters
    # @property
    # def a(self):
    #     return self.raw_a_constraint.transform(self.raw_a)

    # @a.setter
    # def a(self, value):
    #     return self._set_a(value)

    # def _set_a(self, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(self.raw_a)
    #     self.initialize(raw_a=self.raw_a_constraint.inverse_transform(value))

    @property
    def b(self):
        return self.raw_b_constraint.transform(self.raw_b)

    @b.setter
    def b(self, value):
        return self._set_b(value)

    def _set_b(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_b)
        self.initialize(raw_b=self.raw_b_constraint.inverse_transform(value))

    def forward(self, x1, x2=None, last_dim_is_batch=False, diag=False, **params):
        if x2 is None:
            x2 = x1
        # sigmoid = 1/(1+exp(-a(x-b)))
        # `a` is the sharpness of the slope, larger absolute values = sharper slope
        # the sign of `a` determines which side of the curve is 0 and the other is 1
        # `b` is the offset of of the curve at a sigmoid value of 0.5
        x1_ = torch.sigmoid(-self.sharpness * (x1 - self.b))
        x2_ = torch.sigmoid(-self.sharpness * (x2 - self.b))
        
        prod = MatmulLinearOperator(x1_, x2_.transpose(-2, -1))
        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod

class InvertedSigmoidKernel(gpytorch.kernels.Kernel):
    """Inverted sigmoid kernel using a shared SigmoidKernel's parameters.

    K(x1, x2) = (1 - s(x1)) (1 - s(x2))^T, where s(x) is from the provided SigmoidKernel.
    """

    def __init__(self, sigmoid_kernel: SigmoidKernel, active_dims=None, **kwargs):
        super().__init__(active_dims=active_dims, **kwargs)
        self.sigmoid_kernel = sigmoid_kernel

    @property
    def b(self):
        """Delegate b parameter to the original SigmoidKernel."""
        return self.sigmoid_kernel.b

    @b.setter
    def b(self, value):
        self.sigmoid_kernel.b = value

    def forward(self, x1, x2=None, last_dim_is_batch=False, diag=False, **params):
        if x2 is None:
            x2 = x1
        # Compute standard sigmoid using shared b and shared sharpness from base kernel
        sharpness = self.sigmoid_kernel.sharpness
        b = self.sigmoid_kernel.b
        x1_ = torch.sigmoid(-sharpness * (x1 - b))
        x2_ = torch.sigmoid(-sharpness * (x2 - b))
        # Invert: 1 - sigmoid
        x1_inv = 1.0 - x1_
        x2_inv = 1.0 - x2_
        # Outer product for kernel matrix
        prod = MatmulLinearOperator(x1_inv, x2_inv.transpose(-2, -1))
        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod


class LogWarpKernel(gpytorch.kernels.Kernel):
    """
    Wraps a base kernel and applies torch.log(x + eps) to a specified input dimension to avoid log(0).
    """
    def __init__(self, base_kernel, dim, eps=1e-6):
        super().__init__()
        self.base_kernel = base_kernel
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2=None, **params):
        x1_ = x1.clone()
        x1_[:, self.dim] = torch.log(x1_[:, self.dim] + self.eps)
        if x2 is not None:
            x2_ = x2.clone()
            x2_[:, self.dim] = torch.log(x2_[:, self.dim] + self.eps)
        else:
            x2_ = None
        return self.base_kernel(x1_, x2_, **params)


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

# ---- Random Fourier Features ----
class RFFFeatures(torch.nn.Module):
    def __init__(self, in_dim, num_features=128, lengthscale=1.0):
        super().__init__()
        self.register_buffer("omega", torch.randn(in_dim, num_features) / float(lengthscale))
        self.register_buffer("b", 2 * math.pi * torch.rand(num_features))

    def forward(self, x):  # [n,d] -> [n,D]
        proj = x @ self.omega
        return math.sqrt(2.0 / self.omega.shape[1]) * torch.cos(proj + self.b)


# ---- Heteroskedastic noise via RFF + global (positive) scale with a Gamma prior ----
class RFFNoiseModel(gpytorch.Module):
    """Heteroskedastic noise model using Random Fourier Features.

    Produces per-sample noise variances via:
        var(x) = min_noise + noise_scale * softplus(w^T phi(x) + b)

    Parameters are registered following gpytorch conventions (raw_*, constraint, property).
    An optional prior can be attached to the positive ``noise_scale``.

    Args:
        in_dim: Input dimensionality (D).
        num_features: Number of random Fourier features to use.
        min_noise: Non-zero variance floor for stability.
        scale_prior: Optional prior on the positive ``noise_scale``.
        init_scale: Optional initial value for ``noise_scale``.
    """

    def __init__(self, in_dim: int, num_features: int = 128, min_noise: float = 1e-6,
                 scale_prior: GammaPrior | None = None, init_scale: float | None = None):
        super().__init__()
        self.rff = RFFFeatures(in_dim, num_features)
        self.linear = torch.nn.Linear(num_features, 1)
        self.min_noise = float(min_noise)
        self.softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)

        # Global positive scale parameter on the heteroskedastic variance
        self.register_parameter("raw_noise_scale", torch.nn.Parameter(torch.tensor(0.0)))
        self.register_constraint("raw_noise_scale", Positive())

        # Optionally initialize the scale
        if init_scale is not None:
            self._set_noise_scale(torch.as_tensor(init_scale))

        # Initialize to produce near-constant variance initially
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

        if scale_prior is not None:
            self.register_prior(
                "noise_scale_prior",
                scale_prior,
                lambda m: m.noise_scale,
                lambda m, v: m._set_noise_scale(v),
            )

    @property
    def noise_scale(self):
        return self.raw_noise_scale_constraint.transform(self.raw_noise_scale)

    def _set_noise_scale(self, value):
        self.raw_noise_scale.data = self.raw_noise_scale_constraint.inverse_transform(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample variance σ(x)^2.

        Accepts (N, D) or (..., N, D) and returns (N) or (..., N).
        """
        z = self.rff(x)
        raw = self.linear(z).squeeze(-1)
        var = self.noise_scale * self.softplus(raw)
        return var + self.min_noise


# ---- Kernel that adds heteroskedastic variance only on K(X, X) ----
class RFFNoiseKernel(gpytorch.kernels.Kernel):
    def __init__(self, noise_model: RFFNoiseModel):
        super().__init__(has_lengthscale=False)
        self.noise_model = noise_model

    def forward(self, x1, x2=None, diag: bool = False, **params):
        # Treat x2=None as x1
        if x2 is None:
            x2 = x1
        same = (x1.shape == x2.shape) and torch.equal(x1, x2)
        if same:
            d = self.noise_model(x1)
            if diag:
                return d
            return DiagLinearOperator(d)
        else:
            if diag:
                return x1.new_zeros(x1.shape[:-2] + (x1.shape[-2],))
            # Return a low-rank zero operator using only Tensor arguments to satisfy representation constraints
            left = x1.new_zeros(x1.shape[:-2] + (x1.shape[-2], 1))
            right_t = x2.new_zeros(x2.shape[:-2] + (1, x2.shape[-2]))
            return MatmulLinearOperator(left, right_t)