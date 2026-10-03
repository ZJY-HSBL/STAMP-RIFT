# STAMP-RIFT
## STAMP-Informed Risk-Adaptive Interaction-Aware Fuzzy TOPSIS

[中文说明](README.zh-CN.md)

This repository implements an integrated multi-attribute decision framework for competency evaluation in complex emergency-management environments. The research combines systems-theoretic indicator construction, hesitant fuzzy information modeling, sparse criterion interaction learning, and robust TOPSIS ranking within one computational pipeline.

The framework contains four tightly coupled layers:

1. **STAMP-informed indicator construction** for structuring competency criteria from safety constraints, control structures, control actions, and feedback mechanisms.
2. **Risk-adaptive hesitant fuzzy completion**, where each HFE receives a cell-specific completion coefficient according to its hesitation and the decision maker's risk preference.
3. **Sparse 2-additive interaction capacity**, which preserves a target Shapley importance vector while identifying a parsimonious set of complementary or substitutive criterion interactions.
4. **Robust TOPSIS ranking**, including Monte Carlo uncertainty propagation, score intervals, rank acceptability, mean rank, and pairwise superiority probabilities.

## Research contributions

The project focuses on the following methodological contributions:

- a **cell-specific risk-adaptive HFE completion mechanism** that links local hesitation to decision risk preference;
- a **sparse interaction construction strategy** for retaining interpretable criterion complementarity and redundancy while controlling interaction density;
- an **efficient 2-additive Choquet aggregation implementation** suitable for medium- and high-dimensional criterion systems;
- a **robust-ranking layer** that propagates uncertainty in memberships, importance weights, risk parameters, and interaction structure instead of reporting only one deterministic order;
- a unified research workflow that connects **STAMP-based indicator modeling, hesitant fuzzy representation, interaction-aware aggregation, TOPSIS evaluation, and robustness analysis**.

The bundled dataset is **synthetic** and is included to verify the implementation and demonstrate the complete computational workflow. For a domain application, replace it with independently collected case data and a documented expert-evaluation protocol.

## Method summary

For an HFE $h_{ij}$, the hesitation index is defined as

$$
u_{ij}=4\mathrm{Var}(h_{ij}), \qquad u_{ij}\in[0,1].
$$

The effective risk preference and adaptive completion coefficient are

$$
r_{ij}=\mathrm{clip}(r_0+\beta u_{ij},-1,1),
\qquad
\theta_{ij}=\frac{1-r_{ij}}{2}.
$$

The padding value becomes

$$
p_{ij}=\theta_{ij}\max(h_{ij})+(1-\theta_{ij})\min(h_{ij}).
$$

For a 2-additive capacity with Möbius coefficients $m_i$ and $m_{ij}$, the aggregation is

$$
C_\mu(x)=\sum_i m_i x_i+\sum_{i\lt j}m_{ij}\min(x_i,x_j).
$$

The implementation constructs sparse pair interactions from a criterion-association matrix and scales them to satisfy monotonicity while preserving the requested Shapley importance vector.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py demo
```

Evaluate your own case:

```bash
python run.py evaluate \
  --input data/case_template.json \
  --output results/my_case \
  --risk 0.25 \
  --beta 0.5 \
  --interaction-strength 0.15 \
  --interaction-threshold 0.25 \
  --robust 1000
```

Windows example:

```powershell
D:/software/miniconda3/envs/myenv/python.exe -m pip install -r requirements.txt
D:/software/miniconda3/envs/myenv/python.exe -m unittest discover -s tests -v
D:/software/miniconda3/envs/myenv/python.exe run.py demo
```

## Repository structure

```text
stamp_hf/
  hesitant.py      adaptive HFE completion and validation
  capacity.py      lambda and 2-additive fuzzy capacities
  interaction.py   association estimation and sparse interaction construction
  core.py          ideal points, distances and TOPSIS evaluation
  robustness.py    Monte Carlo uncertainty propagation

data/
  synthetic_case.json  transparent synthetic demonstration
  case_template.json   domain-study input template
experiments/
  exp_ablation.py
  exp_robustness.py
tests/
docs/
results/
```

## Research validation

Run the ablation study:

```bash
python experiments/exp_ablation.py
```

Run a 1000-iteration robustness study:

```bash
python experiments/exp_robustness.py
```

The validation workflow supports fixed-vs-adaptive completion comparison, interaction-vs-no-interaction comparison, alternative aggregation models, parameter sensitivity, robustness analysis, rank stability, and computational scalability analysis.

For a domain study, the input should additionally document the indicator system, expert-selection rules, elicitation scale, HFE construction process, criterion directions, importance derivation, association estimation, and robustness parameters.

See [docs/method.md](docs/method.md), [docs/research_intro_en.md](docs/research_intro_en.md), and [docs/research_validation.md](docs/research_validation.md).

## License

MIT.
