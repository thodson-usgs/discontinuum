import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

# Ensure local package is imported before any installed version
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from discontinuum.pipeline import TimeTransformer, LogErrorPipeline


def test_time_transform():
    """ Test the TimeTransformer class.
    """
    # Create sample data
    data = np.array(
        ["2022-01-01", "2022-02-01", "2022-03-01"],
        dtype="datetime64[ns]"  # ns precision
        )
    expected_result = np.array([2022. , 2022.08493151, 2022.16164384])

    # Create TimeTransformer instance
    transformer = TimeTransformer()

    # Perform transform
    transformed_data = transformer.transform(data)

    # Perform inverse transform
    inverse_transformed_data = transformer.inverse_transform(transformed_data)

    # Check if the transformed data matches the expected result
    np.testing.assert_array_almost_equal(transformed_data, expected_result, decimal=6)

    # Check if the inverse transformed data matches the original data
    np.testing.assert_equal(data, inverse_transformed_data)


def test_log_error_pipeline_ci():
    """Ensure LogErrorPipeline.ci computes multiplicative CI correctly."""
    pipeline = LogErrorPipeline()
    mean = 100.0
    se = 0.1
    lower, upper = pipeline.ci(mean, se, ci=0.95)

    alpha = (1 - 0.95) / 2
    zscore = norm.ppf(1 - alpha)
    cb = np.exp(zscore * se)
    expected_lower = mean / cb
    expected_upper = mean * cb

    np.testing.assert_allclose(lower, expected_lower)
    np.testing.assert_allclose(upper, expected_upper)
