import unittest
import numpy as np

from stamp_hf.robustness import robust_evaluate


class RobustnessTests(unittest.TestCase):
    def test_acceptability_rows_sum_to_one(self):
        matrix = [
            [[0.2, 0.3], [0.5, 0.6]],
            [[0.5, 0.6], [0.4, 0.5]],
            [[0.7, 0.8], [0.7, 0.8]],
        ]
        out = robust_evaluate(matrix, [0.5, 0.5], np.eye(2), n_iter=30, seed=4)
        np.testing.assert_allclose(out["rank_acceptability"].sum(axis=1), 1.0)
        self.assertEqual(out["pairwise_superiority"].shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
