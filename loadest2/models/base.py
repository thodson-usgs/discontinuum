import functools


class BaseModel:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def predict(self, X):
        pass


def is_fitted(func):
    """Decorator checks whether model has been fit."""

    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if self.is_fitted:
            raise RuntimeError("The model hasn't been fitted yet, call .fit().")
        return func(self, *args, **kwargs)

    return inner
