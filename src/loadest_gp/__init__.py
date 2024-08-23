import sys
import warnings

__all_gpytorch__ = [
    # Sub-packages in models.gpytorch
    "LoadestGPMarginalGPyTorch",
]

__all_pymc__ = [
    # PyMC sub-packages in models.pymc
    "LoadestGPMarginalPyMC",
]

for name in __all_gpytorch__:
    exec(f"from .models.gpytorch import {name}")

__all__ = __all_gpytorch__.copy()


# check if pymc is installed, then import the sub-packages
try:
    import pymc
    __all__.extend(__all_pymc__)

    for name in __all_pymc__:
        exec(f"from .models.pymc import {name}")

except ModuleNotFoundError:
    # if pymc is not installed, raise a warning
    warnings.warn(
        f"`pymc` is not installed: {', '.join(__all_pymc__)}"
        " will not be available.",
        UserWarning,
    )
