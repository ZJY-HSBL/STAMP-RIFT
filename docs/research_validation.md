# Research validation protocol

The repository separates implementation validation from domain evidence.

The bundled `data/synthetic_case.json` is deterministic synthetic data generated with a fixed seed. It is intended for numerical verification, workflow demonstration, and controlled experiments. Domain conclusions should be derived from independently collected case data.

For a domain study, preserve the following information with the input data or supplementary records:

- criterion definitions and the STAMP mapping used to construct them;
- expert inclusion and exclusion rules, expert count, and expertise profile;
- elicitation scale and evaluation instructions;
- transformation from raw expert responses to hesitant fuzzy elements;
- consensus, reliability, or consistency assessment;
- benefit/cost direction of every criterion;
- derivation or elicitation of criterion importance;
- whether criterion association is externally specified or estimated from data;
- risk, sparsity, interaction-strength, and robustness parameters;
- random seeds and software version.

Recommended validation includes controlled interaction scenarios, fixed-versus-adaptive completion comparison, interaction ablation, parameter sensitivity, perturbation analysis, ranking stability, repeated uncertainty propagation, and scalability experiments. Parameter selection should follow a documented rule rather than being tuned to force a preferred ordering.
