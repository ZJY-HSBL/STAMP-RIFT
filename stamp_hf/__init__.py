"""STAMP and HF multi-attribute correlation TOPSIS."""
from .core import normalize, distances, evaluate, independent_topsis
from .capacity import LambdaCapacity, TwoAdditiveCapacity, decomposition_lp

__all__ = ['normalize', 'distances', 'evaluate', 'independent_topsis',
           'LambdaCapacity', 'TwoAdditiveCapacity', 'decomposition_lp']
