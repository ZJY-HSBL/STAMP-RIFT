# English research introduction

## A STAMP-Based Risk-Adaptive Hesitant Fuzzy Interactive TOPSIS Method for Robust Emergency Competency Evaluation

This study develops an integrated multi-attribute decision framework for competency evaluation in complex emergency-management environments. The method addresses four connected challenges: systematic construction of competency indicators, hesitation in expert judgments, interdependence among evaluation criteria, and uncertainty in final ranking results.

At the indicator-modeling stage, STAMP is used as a systems-theoretic structure for analyzing safety constraints, control structures, control actions, and feedback mechanisms. The resulting analysis is mapped into a hierarchical competency indicator system, providing an explicit systems basis for criterion construction.

Hesitant fuzzy elements preserve multiple possible membership assessments supplied during expert evaluation. A risk-adaptive completion mechanism is introduced in which each HFE receives a cell-specific completion coefficient determined jointly by its local hesitation and a base risk preference. This allows the preprocessing stage to respond to heterogeneous uncertainty across the evaluation matrix.

To represent criterion interdependence, the framework introduces a sparse 2-additive fuzzy capacity. Pair interactions are constructed from a supplied or estimated association structure, sparsified for interpretability, and scaled to satisfy capacity monotonicity while preserving a target Shapley importance vector. Positive and negative interaction coefficients represent complementarity and redundancy among competency dimensions.

The interaction-aware capacity aggregates distances from the positive and negative hesitant fuzzy ideal solutions, after which relative closeness generates the nominal ordering of alternatives. A Monte Carlo uncertainty-propagation layer further perturbs memberships, criterion importance, risk preference, and association structure to estimate score intervals, mean ranks, rank-acceptability probabilities, and pairwise superiority probabilities.

The complete research workflow therefore links STAMP-based indicator modeling, hesitant fuzzy representation, risk-adaptive information completion, sparse interaction identification, 2-additive Choquet aggregation, TOPSIS evaluation, and robust ranking within one unified framework. Validation can be conducted through controlled synthetic experiments and independently collected domain cases, together with ablation, parameter sensitivity, rank-stability, robustness, and computational scalability analyses.
