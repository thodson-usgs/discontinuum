import warnings

from .models.gpytorch import LoadestGPMarginalGPyTorch

__all__ = ["LoadestGPMarginalGPyTorch"]

try:
    import pymc  # noqa: F401
    from .models.pymc import LoadestGPMarginalPyMC  # noqa: F401

    __all__.append("LoadestGPMarginalPyMC")
except ModuleNotFoundError:
    warnings.warn(
        "`pymc` is not installed: LoadestGPMarginalPyMC will not be available.",
        UserWarning,
        stacklevel=2,
    )
