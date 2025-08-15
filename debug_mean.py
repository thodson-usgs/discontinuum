import torch
import gpytorch
import numpy as np
from src.rating_gp.models.gpytorch import ExactGPModel

# Create minimal test case to debug mean function issue
torch.manual_seed(42)

# Training data
train_x = torch.tensor([[0.1, 1.2], [0.3, 1.8], [0.5, 2.1]])
train_y = torch.tensor([0.5, 1.2, 1.8])

# Likelihood
noise = 0.01 * torch.ones(3)
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)

# Model
model = ExactGPModel(train_x, train_y, likelihood)
model.eval()
likelihood.eval()

# Test data
test_x = torch.tensor([[0.2, 1.5]])

print("=== Debugging Mean Function ===")
print(f"Train X: {train_x}")
print(f"Train Y: {train_y}")
print(f"Test X: {test_x}")

with torch.no_grad():
    # Step by step computation
    print("\n--- Step by step computation ---")
    
    # 1. Transform test input
    x_t = test_x.clone()
    stage_raw = x_t[:, model.stage_dim[0]]
    print(f"1. Raw stage: {stage_raw}")
    
    stage_transformed = model.powerlaw(stage_raw)
    x_t[:, model.stage_dim[0]] = stage_transformed
    print(f"2. Transformed stage: {stage_transformed}")
    print(f"3. x_t (transformed input): {x_t}")
    
    # 2. Apply mean function to transformed stage
    mean_direct = model.mean_module(stage_transformed)
    print(f"4. Mean function output: {mean_direct}")
    
    # 3. Covariance on original input
    print(f"5. Covariance input (original x): {test_x}")
    
    # 4. Full model forward
    print("\n--- Full model forward ---")
    output = model(test_x)
    print(f"Model output mean: {output.mean}")
    print(f"Model output variance: {output.variance}")
    
    # 5. Check if they match
    print(f"\n--- Comparison ---")
    print(f"Direct mean: {mean_direct}")
    print(f"Model mean: {output.mean}")
    print(f"Match: {torch.allclose(mean_direct, output.mean, atol=1e-5)}")
    
    # 6. Let's check if the issue is in the GP base class
    print(f"\n--- GP internals ---")
    print(f"Model train_inputs shape: {model.train_inputs[0].shape}")
    print(f"Model train_targets shape: {model.train_targets.shape}")
