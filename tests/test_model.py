import unittest
import numpy as np

from stamp_hf import evaluate, independent_topsis, sparse_capacity


class ModelTests(unittest.TestCase):
    def test_rank_extremes(self):
        matrix = [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
        ]
        cap, _ = sparse_capacity([0.5, 0.5], np.eye(2), strength=0.0)
        result = evaluate(matrix, cap, completion="fixed", theta=0.5)
        np.testing.assert_allclose(result["scores"], [0.0, 1.0])
        np.testing.assert_array_equal(result["order"], [1, 0])

    def test_independent_comparison_model(self):
        matrix = [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
        ]
        np.testing.assert_allclose(independent_topsis(matrix, [0.4, 0.6]), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
