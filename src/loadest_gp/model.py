import numpy as np
import pymc as pm

from discontinuum.data_manager import DataManager
from discontinuum.engines.pymc import MarginalGP
from discontinuum.pipeline import LogStandardPipeline, TimePipeline

from loadest_gp.plot import LoadestPlotMixin

class LoadestGP(MarginalGP, LoadestPlotMixin):
    """
    Gaussian Process implementation of the LOAD ESTimation (LOADEST) model

    This model currrently uses the marginal likelihood implementation, which is fast but does not
    account for censored data. Censored data require a slower latent variable implementation.
    """
    def __init__(self):
        """ """
        super().__init__()
        covariate_pipelines = {"time": TimePipeline, "flow": LogStandardPipeline}
        target_pipeline = LogStandardPipeline

        self.dm = DataManager(
            target_pipeline=target_pipeline, covariate_pipelines=covariate_pipelines
        )

    def build_model(self, X, y) -> pm.Model:
        """ """
        n_covariates = X.shape[1]
        dims = np.arange(n_covariates)
        time_dim = [dims[0]]
        cov_dims = dims[1:]

        with pm.Model() as model:
            # seasonal trend
            eta_per = pm.HalfNormal("eta_per", sigma=1, initval=1)  # was 2
            ls_pdecay = pm.LogNormal("ls_pdecay", mu=2, sigma=1)
            # https://peterroelants.github.io/posts/gaussian-process-kernels/
            period = pm.Normal("period", mu=1, sigma=0.05)
            ls_psmooth = pm.LogNormal("ls_psmooth", mu=1, sigma=1)  # 14 sec at 0,0.5

            cov_seasonal = (
                eta_per**2
                * pm.gp.cov.Periodic(2, period, ls_psmooth, active_dims=time_dim)
                * pm.gp.cov.Matern52(2, ls_pdecay, active_dims=time_dim)
            )
            gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

            # longterm trend
            # eta_trend =  pm.HalfNormal("eta_trend", sigma=1) # was 2
            eta_trend = pm.Exponential(
                "eta_trend", scale=1.5
            )  # Exponential might to limit outliers dictating a trend
            ls_trend = pm.LogNormal("ls_trend", mu=2, sigma=1)
            cov_trend = eta_trend**2 * pm.gp.cov.ExpQuad(2, ls_trend, active_dims=time_dim)
            gp_trend = pm.gp.Marginal(cov_func=cov_trend)

            # covariate trend
            # could include time with a different prior on ls
            eta_covariates = pm.HalfNormal("eta_covariates", sigma=2)
            ls_covariates = pm.LogNormal("ls_covariates", mu=-1.1, sigma=1, initval=0.5)
            cov_covariates = eta_covariates**2 \
                * pm.gp.cov.ExpQuad(2, ls=ls_covariates, active_dims=cov_dims)
            gp_covariates = pm.gp.Marginal(cov_func=cov_covariates)

            # residual trend
            eta_res = pm.Exponential("eta_res", scale=0.2)
            ls_res = pm.LogNormal("ls_res", mu=-1.1, sigma=1, shape=2)
            cov_res = eta_res**2 * pm.gp.cov.ExpQuad(2, ls_res, active_dims=dims)
            gp_res = pm.gp.Marginal(cov_func=cov_res)

            gp = gp_trend + gp_seasonal + gp_covariates + gp_res

            # noise_model
            # set to a small value to ensure cov is positive definite
            # or use a HalfNormal prior
            cov_noise = pm.gp.cov.WhiteNoise(0.1)

            y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=cov_noise)  # noqa: F841

            self.gp = gp

        return model
