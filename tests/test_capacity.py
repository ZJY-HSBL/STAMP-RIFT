import unittest
import numpy as np

from stamp_hf.capacity import LambdaCapacity, TwoAdditiveCapacity
from stamp_hf.interaction import sparse_capacity


class CapacityTests(unittest.TestCase):
    def test_lambda_normalization(self):
        cap = LambdaCapacity([0.2, 0.3])
        self.assertAlmostEqual(cap.value([0, 1]), 1.0, places=9)

    def test_sparse_capacity_preserves_shapley_importance(self):
        w = np.array([0.4, 0.35, 0.25])
        assoc = np.array([[1.0, 0.8, -0.6], [0.8, 1.0, 0.0], [-0.6, 0.0, 1.0]])
        cap, diag = sparse_capacity(w, assoc, strength=0.2, threshold=0.25)
        np.testing.assert_allclose(cap.shapley_importance(), w / w.sum(), atol=1e-10)
        self.assertAlmostEqual(cap.value(range(3)), 1.0, places=9)
        self.assertGreaterEqual(diag["min_marginal"], -1e-10)

    def test_two_additive_integral_constant(self):
        pair = np.array([[0.0, 0.1], [0.1, 0.0]])
        cap = TwoAdditiveCapacity.from_shapley_and_interactions([0.5, 0.5], pair)
        self.assertAlmostEqual(cap.integral([1.0, 1.0]), 1.0)
        self.assertAlmostEqual(cap.integral([0.0, 0.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
