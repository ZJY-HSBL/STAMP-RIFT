"""Generate robust-ranking tables from the transparent synthetic dataset."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import json

from stamp_hf import association_from_matrix, robust_evaluate

data = json.loads((ROOT / "data" / "synthetic_case.json").read_text(encoding="utf-8"))
assoc = association_from_matrix(data["matrix"])
robust = robust_evaluate(
    data["matrix"], data["importance"], assoc,
    benefit=data["benefit"], risk_preference=0.25, beta=0.5,
    n_iter=1000, seed=2026,
)
out = ROOT / "results" / "robustness"
out.mkdir(parents=True, exist_ok=True)
with (out / "rank_acceptability.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["alternative"] + [f"P(rank={i})" for i in range(1, len(data["alternatives"]) + 1)])
    for name, row in zip(data["alternatives"], robust["rank_acceptability"]):
        writer.writerow([name] + row.tolist())
with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["alternative", "mean_score", "ci95_low", "ci95_high", "mean_rank", "p_rank1"])
    for i, name in enumerate(data["alternatives"]):
        writer.writerow([name, robust["mean_score"][i], robust["score_ci95_low"][i], robust["score_ci95_high"][i], robust["mean_rank"][i], robust["rank_acceptability"][i, 0]])
print(out)
