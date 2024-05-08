import pymc as pm

from discontinuum.engines.pymc import PyMCModel


class GPModel(PyMCModel):
    def build_model(self, X, y):
        """ """
        with pm.Model() as self.model:
            # priors
            period = pm.Normal("period", mu=1, sigma=0.05)
            ls_psmooth = pm.Gamma("ls_psmooth", alpha=4, beta=2)
            ls_pdecay = pm.Gamma("ls_pdecay", alpha=4, beta=2)

            ls2 = pm.Gamma("ls2", alpha=4, beta=2)
            eta = pm.HalfNormal("eta", sigma=1)

            gp_time = pm.gp.cov.Matern52(
                1, ls_pdecay, active_dims=[0]
            ) * pm.gp.cov.Periodic(1, period, ls_psmooth, active_dims=[0])

            gp_covariates = pm.gp.cov.ExpQuad(1, ls=ls2, active_dims=[1])

            cov = eta**2 * gp_time * gp_covariates

            gp = pm.gp.Marginal(cov_func=cov)

            # noise
            sigma = pm.HalfCauchy("sigma", beta=1)  # was 0.01

            y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)

            self.gp = gp
