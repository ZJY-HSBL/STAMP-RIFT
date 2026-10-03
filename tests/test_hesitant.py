import unittest
import numpy as np

from stamp_hf.hesitant import adaptive_theta, hesitation_index, normalize_matrix


class HesitantTests(unittest.TestCase):
    def test_hesitation_bounds(self):
        self.assertEqual(hesitation_index([0.5]), 0.0)
        self.assertAlmostEqual(hesitation_index([0.0, 1.0]), 1.0)

    def test_adaptive_completion_is_more_conservative_for_hesitation(self):
        low = adaptive_theta([0.49, 0.51], risk_preference=0.0, beta=0.8)
        high = adaptive_theta([0.0, 1.0], risk_preference=0.0, beta=0.8)
        self.assertLess(high, low)

    def test_cost_transform(self):
        x = normalize_matrix([[[0.2, 0.4]]], benefit=[False], completion="fixed", theta=0.5)
        np.testing.assert_allclose(x[0, 0], [0.6, 0.8])


if __name__ == "__main__":
    unittest.main()
