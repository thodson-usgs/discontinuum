import gpytorch
import numpy as np
import torch

from linear_operator.operators import MatmulLinearOperator, to_dense


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
    """Sigmoid Kernel

    This kernel can be multiplied by another kernel to give a breakpoint in the
    data using a sigmoid

    Note: The strength of the slope of the sigmoid is currently fixed to a steep
    slope. This can be turned back into a parameter, but it cause numerical
    instabilities during fitting.
    """
    def __init__(
        self,
        # a_prior=None,
        # a_constraint=None,
        b_constraint,
        b_prior=None,
        #b_constraint=gpytorch.constraints.Positive(),
        **kwargs,
        ):
        """Initialize the kernel

        Parameters
        ----------
        b_prior : Prior
            The prior to impose on `b`.
        b_constraint : Constraint
            The constraint to impose on `b`
        """
        super().__init__(**kwargs)

        self.a = 20
        # self.a = torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1) * 40)
        # self.register_parameter(
        #     name='raw_a',
        #     parameter=self.a)
        
        u = b_constraint.upper_bound
        l = b_constraint.lower_bound
        self.b = torch.nn.Parameter(
            l + torch.rand((*self.batch_shape, 1, 1)) * (u - l)
            #0.5 - torch.rand(*self.batch_shape, 1, 1)
        )

        self.register_parameter(
            name='raw_b',
            # the changepoint is in log-standardized q, so initialize with randn
            # set the b as a rand in the rand 
            parameter=self.b
        )

        b_prior = gpytorch.priors.NormalPrior(0, 1)

        #self.register_prior(
        #    "b_prior",
        #    gpytorch.priors.NormalPrior(0, 1),
        #    lambda module: module.b,
        #)

        # register the constraints
        # if a_constraint is None:
        #     a_constraint = gpytorch.constraints.Positive()
        # self.register_constraint("raw_a", a_constraint)
        if b_constraint is not None:
            self.register_constraint("raw_b", b_constraint)

        # set the parameter prior
        # if a_prior is not None:
        #     self.register_prior(
        #         "a_prior",
        #         a_prior,
        #         lambda m: m.a,
        #         lambda m, v : m._set_a(v),
        #     )
        if b_prior is not None:
            self.register_prior(
                "b_prior",
                b_prior,
                lambda m: m.b,
                lambda m, v : m._set_b(v),
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

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        # sigmoid = 1/(1+exp(-a(x-b)))
        # `a` is the sharpness of the slope, larger absolute values = sharper slope
        # the sign of `a` determines which side of the curve is 0 and the other is 1
        # `b` is the offset of of the curve at a sigmoid value of 0.5
        x1_ = 1/(1 + torch.exp(self.a * (x1 - self.b)))
        x2_ = 1/(1 + torch.exp(self.a * (x2 - self.b)))
        
        prod = MatmulLinearOperator(x1_, x2_.transpose(-2, -1))
        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod

# Inverted sigmoid kernel as a proper GPyTorch kernel
class InvertedSigmoidKernel(SigmoidKernel):
    def __init__(self, sigmoid_kernel, active_dims=None, b_constraint=None):
        # Initialize without its own raw_b; will share b parameter via sigmoid_kernel
        super().__init__(active_dims=active_dims, b_constraint=b_constraint)
        self.sigmoid_kernel = sigmoid_kernel

    # @property
    # def a(self):
    #     """Delegate a parameter to the original SigmoidKernel."""
    #     return self.sigmoid_kernel.a

    # @a.setter
    # def a(self, value):
    #     self.sigmoid_kernel.a = value 

    @property
    def b(self):
        """Delegate b parameter to the original SigmoidKernel."""
        return self.sigmoid_kernel.b

    @b.setter
    def b(self, value):
        self.sigmoid_kernel.b = value

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        # Compute standard sigmoid using shared b
        x1_ = 1/(1 + torch.exp(self.a * (x1 - self.b)))
        x2_ = 1/(1 + torch.exp(self.a * (x2 - self.b)))
        # Invert: 1 - sigmoid
        x1_inv = 1.0 - x1_
        x2_inv = 1.0 - x2_
        # Outer product for kernel matrix
        prod = MatmulLinearOperator(x1_inv, x2_inv.transpose(-2, -1))
        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod
