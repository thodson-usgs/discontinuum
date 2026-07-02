import torch

from discontinuum.noise import OrthogonalRFFNoise


def test_per_dimension_scaling():
    input_dim = 3
    num_features = 16
    scales = torch.tensor([2.0, 0.5, 1.5])
    x = torch.randn(10, input_dim)

    model_scaled = OrthogonalRFFNoise(
        input_dim=input_dim,
        num_features=num_features,
        scales=scales,
        seed=0,
    )
    features_scaled = model_scaled(x)

    # Equivalent computation by scaling the inputs instead of providing scales
    model_unscaled = OrthogonalRFFNoise(
        input_dim=input_dim,
        num_features=num_features,
        seed=0,
    )
    features_unscaled = model_unscaled(x / scales)

    assert torch.allclose(features_scaled, features_unscaled, atol=1e-5)
