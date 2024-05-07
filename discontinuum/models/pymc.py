import pymc as pm

from .base import BaseModel


class PyMCModel(BaseModel):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def predict(self, X):
        pass


def init_gp(
    X,
    y,
    white_noise=True,
    censored=False,
):
    """ """
    ls_pdecay = pm.Gamma("ℓ_pdecay", alpha=4, beta=2)
    period = pm.Normal("period", mu=1, sigma=0.05)
    ls_psmooth = pm.Gamma("ls_psmooth", alpha=4, beta=2)
    eta_per = pm.HalfNormal("eta_per", sigma=2)  # was 2
    cov_seasonal = (
        eta_per**2
        * pm.gp.cov.Periodic(1, period, ls_psmooth, active_dims=[1])
        * pm.gp.cov.Matern52(1, ls_pdecay, active_dims=[1])
    )

    # flow trend
    ls2 = pm.Gamma("ls2", alpha=4, beta=2)
    eta_flow = pm.HalfNormal("eta_flow", sigma=2)  # was 2
    cov_flow = eta_flow**2 * pm.gp.cov.ExpQuad(1, ls=ls2, active_dims=[0])

    # noise
    if white_noise is True:
        sigma_noise = pm.HalfCauchy("sigma", beta=1)
        cov_noise = pm.gp.cov.WhiteNoise(sigma_noise)

    if not censored:
        # use the marginal likelihood
        gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)
        gp_flow = pm.gp.Marginal(cov_func=cov_flow)
        gp = gp_seasonal + gp_flow
        y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=cov_noise)

    else:
        # use a latent
        gp_seasonal = pm.gp.Latent(cov_func=cov_seasonal)
        gp_flow = pm.gp.Latent(cov_func=cov_flow)
        gp = gp_seasonal + gp_flow
        f = gp.prior("f", X=X)
        y_ = pm.Normal("y", mu=f, sigma=cov_noise, observed=y)


def init_muli_gp(
    X,
    y,
    white_noise=True,
    censored=False,
):
    """Don't log transform data"""
    # Set priors on the hyperparameters of the covariance
    # ls1 = pm.Gamma("ls1", alpha=4, beta=2) #all gammas were 4,2 in 60 sec
    period = pm.Normal("period", mu=1, sigma=0.05)
    # ls_psmooth = pm.Gamma("ls_psmooth", alpha=4, beta=2) # 11 seconds
    ls_psmooth = pm.Gamma("ls_psmooth", alpha=4, beta=2)  # TODO test against above
    # ls_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075) #58 seconds
    ls_pdecay = pm.Gamma("ℓ_pdecay", alpha=4, beta=2)  # was 4,2
    ls2 = pm.Gamma("ls2", alpha=4, beta=2)

    gp1 = pm.gp.cov.Matern52(1, ls_pdecay, active_dims=[1]) * pm.gp.cov.Periodic(
        1, period, ls_psmooth, active_dims=[1]
    )

    gp2 = pm.gp.cov.ExpQuad(1, ls=ls2, active_dims=[0])

    eta = pm.HalfNormal("eta", sigma=1)  # was 2

    cov = eta**2 * gp1 * gp2

    if white_noise:
        cov_noise = pm.HalfCauchy("sigma", beta=1)
    else:
        raise NotImplementedError

    if not censored:
        # use the marginal likelihood
        gp = pm.gp.Marginal(cov_func=cov)
        y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=cov_noise)

    elif censored:
        # use a latent
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=X)
        y_ = pm.Normal("y", mu=f, sigma=cov_noise, observed=y)
