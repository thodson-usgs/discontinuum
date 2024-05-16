import pymc as pm

from discontinuum.engines.pymc import PyMCModel
from discontinuum.data_manager import DataManager
from discontinuum.pipeline import LogStandardPipeline, TimePipeline
from discontinuum.models.wrtds.plot import PlotMixin


class WRTDSModel(PyMCModel, PlotMixin):
    def __init__(self):
        """ """
        super().__init__()
        covariate_pipelines = {'time': TimePipeline, 'flow': LogStandardPipeline}
        target_pipeline = LogStandardPipeline

        self.dm = DataManager(
            target_pipeline=target_pipeline, covariate_pipelines=covariate_pipelines
        )

    def build_model(self, X, y) -> pm.Model:
        """ """
        with pm.Model() as model:
            # seasonal trend
            eta_per = pm.HalfNormal("eta_per", sigma=1, initval=1)  # was 2
            ls_pdecay = pm.LogNormal("ls_pdecay", mu=2, sigma=1)
            # https://peterroelants.github.io/posts/gaussian-process-kernels/
            period = pm.Normal("period", mu=1, sigma=0.05)
            ls_psmooth = pm.LogNormal("ls_psmooth", mu=1, sigma=1)  # 14 sec at 0,0.5

            cov_seasonal = (
                eta_per**2
                * pm.gp.cov.Periodic(2, period, ls_psmooth, active_dims=[0])
                * pm.gp.cov.Matern52(2, ls_pdecay, active_dims=[0])
            )
            gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

            # longterm trend
            # eta_trend =  pm.HalfNormal("eta_trend", sigma=1) # was 2
            eta_trend = pm.Exponential(
                "eta_trend", scale=1.5
            )  # Exponential might to limit outliers dictating a trend
            ls_trend = pm.LogNormal("ls_trend", mu=2, sigma=1)
            cov_trend = eta_trend**2 * pm.gp.cov.ExpQuad(2, ls_trend, active_dims=[0])
            gp_trend = pm.gp.Marginal(cov_func=cov_trend)

            # flow trend
            eta_flow = pm.HalfNormal("eta_flow", sigma=2)  # was 2
            ls_flow = pm.LogNormal("ls_flow", mu=-1.1, sigma=1, initval=0.5)
            cov_flow = eta_flow**2 * pm.gp.cov.ExpQuad(2, ls=ls_flow, active_dims=[1])
            gp_flow = pm.gp.Marginal(cov_func=cov_flow)

            # noise model
            eta_noise = pm.Exponential(
                "eta_noise", scale=0.2
            )  # 0.5 worked good for chop: 720 and 1:17
            ls_noise = pm.LogNormal("ls_noise", mu=-1.1, sigma=1, shape=2)
            # ls_cov_noise = pm.LogNormal("ls_cov_noise", mu=-1.1, sigma=1, initval=0.1)
            cov_noise = eta_noise**2 * pm.gp.cov.ExpQuad(
                2, ls_noise, active_dims=[0, 1]
            )
            gp_noise = pm.gp.Marginal(cov_func=cov_noise)

            gp = gp_trend + gp_seasonal + gp_noise + gp_flow

            # noise_model
            cov_measure = pm.gp.cov.WhiteNoise(0.1)  # 40 seconds with  eta_noise = 0.1

            y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=cov_measure)

            self.gp = gp

        return model
