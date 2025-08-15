#!/usr/bin/env python3
"""Test script to verify the resume functionality works correctly."""

import torch
from discontinuum.engines.gpytorch import MarginalGPyTorch


# Create a dummy model class (avoid pytest treating it as a test class)
class DummyModel(MarginalGPyTorch):
    def build_model(self, X, y, **kwargs):
        import gpytorch

        class SimpleGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SimpleGP(X, y, self.likelihood)
        return model


print("Testing iteration tracking...")
model = DummyModel()
print(f"Initial iteration: {model._current_iteration}")

model._current_iteration = 50
print(f"After setting to 50: {model._current_iteration}")
