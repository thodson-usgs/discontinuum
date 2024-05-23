import numpy as np
from discontinuum.pipeline import TimeTransformer


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
