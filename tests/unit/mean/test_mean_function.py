#!/usr/bin/env python3
"""
Test script to verify the mean function is working correctly in the Rating GP model.
"""
import torch
import numpy as np
from discontinuum.engines.gpytorch import NoOpMean
from rating_gp.models.gpytorch import PowerLawTransform, ExactGPModel
import gpytorch

def test_noop_mean():
    """Test that NoOpMean correctly returns input squeezed to remove last dimension."""
    mean_module = NoOpMean()
    x1 = torch.tensor([1.0, 2.0, 3.0])
    result1 = mean_module(x1)
    assert torch.allclose(x1, result1)

    x2 = torch.tensor([[1.0], [2.0], [3.0]])
    result2 = mean_module(x2)
    assert result2.shape == (3,)
    assert torch.allclose(x2.squeeze(-1), result2)


def test_powerlaw_transform():
    powerlaw = PowerLawTransform()
    stage_values = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
    result = powerlaw(stage_values)
    # gradient flow check
    loss = result.sum()
    loss.backward()
    assert powerlaw.a.grad is not None
    assert powerlaw.c.grad is not None


def test_exactgp_forward():
    n_points = 20
    torch.manual_seed(42)
    train_x = torch.rand(n_points, 2)
    train_x[:, 1] = train_x[:, 1] * 2 + 1
    train_y = torch.randn(n_points)
    noise = 0.1**2 * torch.ones(n_points)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
    model = ExactGPModel(train_x, train_y, likelihood)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        test_x = torch.tensor([[0.5, 1.5], [0.7, 2.0], [0.3, 2.5]])
        output = model(test_x)
        assert output.mean.shape[0] == test_x.shape[0]
