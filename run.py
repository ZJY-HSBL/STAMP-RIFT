"""Command-line entry point for the research framework."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stamp_hf import (
    association_from_matrix,
    evaluate,
    independent_topsis,
    robust_evaluate,
    sparse_capacity,
)

ROOT = Path(__file__).resolve().parent


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def load_dataset(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["alternatives", "criteria", "matrix", "importance"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if len(data["alternatives"]) != len(data["matrix"]):
        raise ValueError("alternative labels must match matrix rows")
    if len(data["criteria"]) != len(data["importance"]):
        raise ValueError("criteria and importance lengths differ")
    return data


def build_capacity(data, strength, threshold):
    association = data.get("association")
    if association is None:
        association = association_from_matrix(data["matrix"])
    cap, diagnostics = sparse_capacity(
        data["importance"],
        association,
        strength=strength,
        threshold=threshold,
    )
    return cap, np.asarray(association, dtype=float), diagnostics


def evaluate_case(data, output, risk, beta, strength, threshold, robust_iterations, seed):
    output.mkdir(parents=True, exist_ok=True)
    cap, association, diagnostics = build_capacity(data, strength, threshold)
    result = evaluate(
        data["matrix"],
        cap,
        benefit=data.get("benefit"),
        completion="adaptive",
        risk_preference=risk,
        beta=beta,
    )
    comparison = independent_topsis(
        data["matrix"],
        data["importance"],
        benefit=data.get("benefit"),
        completion="adaptive",
        risk_preference=risk,
        beta=beta,
    )
    names = data["alternatives"]
    ranking = [names[i] for i in result["order"]]

    payload = {
        "method": "risk-adaptive hesitant fuzzy interactive TOPSIS",
        "provenance": data.get("provenance", "user-supplied dataset"),
        "parameters": {
            "risk_preference": risk,
            "beta": beta,
            "interaction_strength": strength,
            "interaction_threshold": threshold,
        },
        "interaction_diagnostics": diagnostics,
        "shapley_importance": cap.shapley_importance(),
        "interaction_index": cap.interaction_index(),
        "association": association,
        "scores": result["scores"],
        "ranking": ranking,
        "independent_topsis_scores": comparison,
    }

    if robust_iterations:
        robust = robust_evaluate(
            data["matrix"],
            data["importance"],
            association,
            benefit=data.get("benefit"),
            risk_preference=risk,
            beta=beta,
            interaction_strength=strength,
            interaction_threshold=threshold,
            n_iter=robust_iterations,
            seed=seed,
        )
        payload["robustness"] = {k: v for k, v in robust.items() if not k.endswith("_samples")}

    save_json(output / "result.json", payload)
    with (output / "ranking.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        header = ["position", "alternative", "score", "independent_score"]
        if "robustness" in payload:
            header += ["mean_score", "ci95_low", "ci95_high", "mean_rank", "p_rank1"]
        writer.writerow(header)
        robust = payload.get("robustness")
        for pos, idx in enumerate(result["order"], 1):
            row = [pos, names[idx], result["scores"][idx], comparison[idx]]
            if robust:
                row += [
                    robust["mean_score"][idx],
                    robust["score_ci95_low"][idx],
                    robust["score_ci95_high"][idx],
                    robust["mean_rank"][idx],
                    robust["rank_acceptability"][idx][0],
                ]
            writer.writerow(row)
    print(" > ".join(ranking))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the bundled transparent synthetic study")
    demo.add_argument("--output", type=Path, default=ROOT / "results" / "demo")
    demo.add_argument("--robust", type=int, default=300, help="Monte Carlo iterations; 0 disables")
    demo.add_argument("--seed", type=int, default=2026)

    evaluate_p = sub.add_parser("evaluate", help="evaluate a JSON case study")
    evaluate_p.add_argument("--input", type=Path, required=True)
    evaluate_p.add_argument("--output", type=Path, default=ROOT / "results" / "case")
    evaluate_p.add_argument("--risk", type=float, default=0.25)
    evaluate_p.add_argument("--beta", type=float, default=0.5)
    evaluate_p.add_argument("--interaction-strength", type=float, default=0.15)
    evaluate_p.add_argument("--interaction-threshold", type=float, default=0.25)
    evaluate_p.add_argument("--robust", type=int, default=500)
    evaluate_p.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()
    try:
        if args.command == "demo":
            data = load_dataset(ROOT / "data" / "synthetic_case.json")
            evaluate_case(data, args.output, 0.25, 0.5, 0.15, 0.25, args.robust, args.seed)
        else:
            data = load_dataset(args.input)
            evaluate_case(
                data,
                args.output,
                args.risk,
                args.beta,
                args.interaction_strength,
                args.interaction_threshold,
                args.robust,
                args.seed,
            )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Input or computation error: {exc}\n")


if __name__ == "__main__":
    main()
