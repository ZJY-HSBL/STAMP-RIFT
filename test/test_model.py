import json
from pathlib import Path
import unittest
import numpy as np
from stamp_hf import LambdaCapacity, TwoAdditiveCapacity, decomposition_lp, evaluate, normalize, distances
from stamp_hf.core import independent_topsis

class NumericalTests(unittest.TestCase):
    def test_padding_and_cost(self):
        x=normalize([[[.2,.4]],[[.1,.5,.9]]],theta=.5)
        np.testing.assert_allclose(x[0,0],[.2,.3,.4])
        np.testing.assert_allclose(normalize([[[.2,.4]]],benefit=[False])[0,0],[.6,.8])

    def test_paper_ideals(self):
        p=json.loads((Path(__file__).resolve().parents[1]/'data/paper_public.json').read_text())
        pos,neg,dp,dm=distances(normalize(p['matrix']))
        np.testing.assert_allclose(pos[0],[.3,.3,.3,.5,.8])
        np.testing.assert_allclose(neg[3],[.7,.7,.7,.7,.8])
        self.assertEqual(dp[1,0],0)
        self.assertAlmostEqual(dp[0,0],np.sqrt(.28/5))
        self.assertEqual(dm[0,0],0)

    def test_lambda_normalization(self):
        for s in [[.2,.3],[.6,.7],[.3,.7],[.02]*26]:
            c=LambdaCapacity(s)
            self.assertAlmostEqual(c.value(range(c.n)),1,places=9)
            for i in range(c.n): self.assertAlmostEqual(c.value([i]),s[i])
        self.assertAlmostEqual(LambdaCapacity([.2,.3]).lam,.5/.06)
        self.assertLess(LambdaCapacity([.6,.7]).lam,0)

    def test_fast_integral_matches_exhaustive_lp(self):
        rng=np.random.default_rng(43)
        for n in [2,3,5,7]:
            for total in [.6,1.,1.3]:
                s=rng.uniform(.5,1,size=n); s=s/s.sum()*total
                cap=LambdaCapacity(s)
                for _ in range(4):
                    f=rng.uniform(0,1,n)
                    self.assertAlmostEqual(cap.integral(f),decomposition_lp(f,cap),places=8)

    def test_two_additive_lp(self):
        for interaction in [.2,-.2]:
            c=TwoAdditiveCapacity([.5,.5],[[0,interaction],[interaction,0]])
            self.assertAlmostEqual(c.integral([.2,.8]),decomposition_lp([.2,.8],c))
        c=TwoAdditiveCapacity([1/3]*3,[[0,.1,-.1],[.1,0,0],[-.1,0,0]])
        self.assertAlmostEqual(c.value(range(3)),1)
        self.assertAlmostEqual(c.integral([.2,.7,.4]),decomposition_lp([.2,.7,.4],c))

    def test_rank_and_ties(self):
        c=LambdaCapacity([.2,.3])
        r=evaluate([[[0],[0]],[[1],[1]]],c)
        np.testing.assert_allclose(r['scores'],[0,1])
        np.testing.assert_array_equal(r['order'],[1,0])
        r=evaluate([[[.5],[.5]],[[.5],[.5]]],c)
        np.testing.assert_allclose(r['scores'],[.5,.5])
        np.testing.assert_array_equal(r['order'],[0,1])
        np.testing.assert_allclose(independent_topsis([[[0],[0]],[[1],[1]]],[.4,.6]),[0,1])

    def test_integral_properties(self):
        cap=LambdaCapacity([.15,.2,.1]); f=np.array([.2,.5,.7])
        self.assertAlmostEqual(cap.integral(f*2),cap.integral(f)*2)
        self.assertLessEqual(cap.integral(f),cap.integral(f+.1))
        self.assertAlmostEqual(cap.integral(np.ones(3)),1)
        self.assertEqual(cap.integral(np.zeros(3)),0)

    def test_invalid_inputs(self):
        for matrix in [[],[[[]]],[[[float('nan')]]],[[[1.1]]],[[[.5]],[[.2],[.3]]]]:
            with self.assertRaises(ValueError): normalize(matrix)
        with self.assertRaises(ValueError): normalize([[[.5]]],theta=2)
        with self.assertRaises(ValueError): LambdaCapacity([0,.2])
        with self.assertRaises(ValueError): TwoAdditiveCapacity([.5,.5],[[0,1.1],[1.1,0]])
        with self.assertRaises(ValueError): decomposition_lp([1]*26,LambdaCapacity([.02]*26))
        with self.assertRaises(ValueError): evaluate([[[.2]]],LambdaCapacity([.2,.3]))

if __name__=='__main__': unittest.main()
