import unittest

import numpy as np

from discontinuum.pipeline import TimeTransformer


class TimeTransformerTest(unittest.TestCase):
    def test_time_transform(self):
        # Create sample data
        data = np.array(["2022-01-01", "2022-02-01", "2022-03-01"], dtype="datetime64")
        expected_result = np.array([2022.0, 2022.084, 2022.167])

        # Create TimeTransformer instance
        transformer = TimeTransformer()

        # Perform transform
        transformed_data = transformer.transform(data)

        # Perform inverse transform
        inverse_transformed_data = transformer.inverse_transform(transformed_data)

        # Check if the transformed data matches the expected result
        np.testing.assert_array_almost_equal(inverse_transformed_data, expected_result)

        # Check if the inverse transformed data matches the original data
        np.testing.assert_equal(data, inverse_transformed_data)


if __name__ == "__main__":
    unittest.main()