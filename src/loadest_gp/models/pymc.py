import numpy as np
import pymc as pm
from discontinuum.engines.pymc import MarginalPyMC

from loadest_gp.models.base import LoadestDataMixin, ModelConfig
from loadest_gp.plot import LoadestPlotMixin


class LoadestGPMarginalPyMC(LoadestDataMixin, LoadestPlotMixin, MarginalPyMC):
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
        super(MarginalPyMC, self).__init__(model_config=model_config)
        self.build_datamanager(model_config)

    def build_model(self, X, y) -> pm.Model:
        """Build marginal likelihood version of LoadestGP
        """
        n_d = X.shape[1]  # number of dimensions
        dims = np.arange(n_d)
        time_dim = [dims[0]]
        cov_dims = dims[1:]

        with pm.Model() as model:
            # seasonal trend
            eta_per = pm.HalfNormal("eta_per", sigma=1, initval=1)
            ls_pdecay = pm.Gamma("ls_pdecay", alpha=10, beta=1)
            # https://peterroelants.github.io/posts/gaussian-process-kernels/
            period = pm.Normal("period", mu=1, sigma=0.05)
            ls_psmooth = pm.Gamma("ls_psmooth", alpha=4, beta=3)

            cov_seasonal = (
                eta_per**2
                * pm.gp.cov.Periodic(n_d, period, ls_psmooth, active_dims=time_dim)
                * pm.gp.cov.Matern52(n_d, ls_pdecay, active_dims=time_dim)
            )
            gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

            # longterm trend
            eta_trend = pm.Exponential(
                "eta_trend", scale=1.5
            )  # Exponential might dampen outlier effects
            ls_trend = pm.Gamma("ls_trend", alpha=4, beta=1)
            cov_trend = eta_trend**2 * pm.gp.cov.ExpQuad(n_d, ls_trend, active_dims=time_dim)
            gp_trend = pm.gp.Marginal(cov_func=cov_trend)

            # covariate trend
            # could include time with a different prior on ls
            eta_covariates = pm.HalfNormal("eta_covariates", sigma=2)
            ls_covariates = pm.Gamma(
                "ls_covariates",
                alpha=2,
                beta=3,
                initval=[0.5],
                shape=n_d-1,  # exclude time
                )
            cov_covariates = eta_covariates**2 \
                * pm.gp.cov.ExpQuad(n_d, ls=ls_covariates, active_dims=cov_dims)
            gp_covariates = pm.gp.Marginal(cov_func=cov_covariates)

            # residual trend
            eta_res = pm.Exponential("eta_res", scale=0.2)
            ls_res = pm.Gamma("ls_res", alpha=2, beta=10, shape=n_d)
            cov_res = eta_res**2 * pm.gp.cov.Matern32(n_d, ls_res, active_dims=dims)
            gp_res = pm.gp.Marginal(cov_func=cov_res)

            gp = gp_trend + gp_seasonal + gp_covariates + gp_res

            # noise_model
            # set to a small value to ensure cov is positive definite
            sigma = 0.1
            # or
            # sigma = pm.HalfNormal("sigma", sigma=0.05)
            cov_noise = pm.gp.cov.WhiteNoise(sigma=sigma)

            y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=cov_noise)  # noqa: F841

            self.gp = gp

        return model
