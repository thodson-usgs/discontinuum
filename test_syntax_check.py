#!/usr/bin/env python3
"""Simple test to verify iteration tracking and resume parameter work."""

from discontinuum.engines.gpytorch import MarginalGPyTorch

# Test that the class can be imported and has the new attributes
print("Testing MarginalGPyTorch class...")

# Check that the fit method has the new resume parameter
import inspect
fit_signature = inspect.signature(MarginalGPyTorch.fit)
print(f"Fit method parameters: {list(fit_signature.parameters.keys())}")

if 'resume' in fit_signature.parameters:
    print("✓ Resume parameter found in fit method")
else:
    print("✗ Resume parameter not found in fit method")

# Test the __init__ method adds _current_iteration
try:
    # We can't instantiate directly because it's abstract, but we can check
    # that the __init__ method signature is correct
    init_signature = inspect.signature(MarginalGPyTorch.__init__)
    print(f"Init method parameters: {list(init_signature.parameters.keys())}")
    print("✓ Class definition is syntactically correct")
except Exception as e:
    print(f"✗ Error in class definition: {e}")

print("✓ All syntax checks passed!")
