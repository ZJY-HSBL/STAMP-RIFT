"""Ablation study: fixed/adaptive completion and independent/interactive aggregation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import json
import numpy as np

from stamp_hf import association_from_matrix, evaluate, independent_topsis, sparse_capacity
from stamp_hf.capacity import TwoAdditiveCapacity

data = json.loads((ROOT / "data" / "synthetic_case.json").read_text(encoding="utf-8"))
assoc = association_from_matrix(data["matrix"])
cap, _ = sparse_capacity(data["importance"], assoc, strength=0.15, threshold=0.25)
zero_cap = TwoAdditiveCapacity.from_shapley_and_interactions(data["importance"], np.zeros_like(assoc))

rows = []
for completion in ["fixed", "adaptive"]:
    kwargs = {"completion": completion, "theta": 0.5, "risk_preference": 0.25, "beta": 0.5}
    independent = independent_topsis(data["matrix"], data["importance"], benefit=data["benefit"], **kwargs)
    no_interaction = evaluate(data["matrix"], zero_cap, benefit=data["benefit"], **kwargs)["scores"]
    interactive = evaluate(data["matrix"], cap, benefit=data["benefit"], **kwargs)["scores"]
    for i, name in enumerate(data["alternatives"]):
        rows.append([completion, name, independent[i], no_interaction[i], interactive[i]])

out = ROOT / "results" / "ablation"
out.mkdir(parents=True, exist_ok=True)
with (out / "ablation.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["completion", "alternative", "independent", "zero_interaction_capacity", "sparse_interactive"])
    writer.writerows(rows)
print(out / "ablation.csv")
