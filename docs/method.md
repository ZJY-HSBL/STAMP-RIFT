# Method definition

## 1. Problem setting

Let there be alternatives $A=\{a_1,\ldots,a_m\}$ and criteria $C=\{c_1,\ldots,c_n\}$. Each evaluation $h_{ij}$ is a hesitant fuzzy element containing one or more membership values in $[0,1]$.

The present implementation treats STAMP as an **indicator-construction methodology**, not an automatic text-to-indicator algorithm. The numerical model begins once a validated criterion system, HFE decision matrix, benefit/cost flags and criterion importance vector have been supplied.

## 2. Risk-adaptive HFE completion

Different HFEs may contain different numbers of values. Let

$$
u_{ij}=4\mathrm{Var}(h_{ij}).
$$

Because values lie in $[0,1]$, $u_{ij}\in[0,1]$. Given base risk preference $r_0\in[-1,1]$ and hesitation sensitivity $\beta\ge0$, define

$$
r_{ij}=\mathrm{clip}(r_0+\beta u_{ij},-1,1)
$$

and

$$
\theta_{ij}=(1-r_{ij})/2.
$$

If an HFE needs padding, the inserted value is

$$
p_{ij}=\theta_{ij}\max(h_{ij})+(1-\theta_{ij})\min(h_{ij}).
$$

Positive risk preference therefore shifts the completion toward the minimum membership value.

## 3. Criterion association and sparse interactions

When an external interaction matrix is unavailable, criterion association is estimated from the alternative-level mean membership scores using Spearman correlation. This is an empirical association estimate, not a causal claim.

For target Shapley importance $w_i$, a thresholded association matrix produces preliminary pair interactions $q_{ij}$. A global factor $\gamma\in[0,1]$ is selected so that

$$
m_i=w_i-\frac12\sum_{j\ne i}\gamma q_{ij}
$$

and the compact monotonicity condition

$$
m_i+\sum_{j:m_{ij}\lt 0}m_{ij}\ge0
$$

holds for every criterion. The resulting 2-additive capacity is normalized automatically because the Shapley vector sums to one.

## 4. 2-additive Choquet aggregation

For non-negative vector $x$,

$$
C_\mu(x)=\sum_i m_i x_i+\sum_{i\lt j}m_{ij}\min(x_i,x_j).
$$

This avoids enumeration of $2^n$ subsets and remains practical for dozens of criteria.

## 5. TOPSIS stage

After HFE completion and benefit/cost normalization, positive and negative hesitant fuzzy ideal points are obtained coordinate-wise. Per-criterion RMS distances are aggregated by the selected capacity:

$$
Z_i^+=C_\mu(d_i^+),\qquad Z_i^-=C_\mu(d_i^-).
$$

The closeness score is

$$
S_i=\frac{Z_i^-}{Z_i^+ + Z_i^-}.
$$

Higher $S_i$ indicates a better alternative.

## 6. Robust ranking

The robustness module perturbs HFE memberships, importance weights, association values and risk preference. For each simulation it rebuilds a valid sparse capacity and recomputes the complete ranking. It reports:

- mean score and 95% empirical score interval;
- mean rank;
- rank acceptability $P(R_i=r)$;
- pairwise superiority $P(S_i\gt S_j)$.

These quantities describe ranking stability rather than replacing domain validation.
