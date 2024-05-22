from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional

import functools


class BaseModel:
    def __init__(self, model_config: Optional[Dict] = None):
        """ """
        if model_config is None:
            model_config = {}

        self.dm = None
        self.model_config = model_config
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
        if not self.is_fitted:
            raise RuntimeError(
                "The model hasn't been fitted yet, call .fit()."
            )
        return func(self, *args, **kwargs)

    return inner
