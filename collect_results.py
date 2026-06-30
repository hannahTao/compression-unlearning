"""
Collect TOFU_SUMMARY.json files from harness saves/ into a single results CSV.
Usage: python collect_results.py
"""
import json
import csv
import os
from pathlib import Path

HARNESS_DIR = Path("/workspace/open-unlearning")
OUT_CSV = Path("/workspace/compression-unlearning/baselines.csv")

METRICS = ["forget_Q_A_Prob", "forget_Q_A_ROUGE", "extraction_strength", "model_utility"]

RUNS = {
    # anchors (pre-computed by setup_data.py)
    "ceiling_full":   "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10",
    "floor_retain90": "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90",
    # step-2 baselines
    "baseline_NPO":    "saves/eval/baseline_NPO",
    "baseline_SimNPO": "saves/eval/baseline_SimNPO",
    "baseline_IdkDPO": "saves/eval/baseline_IdkDPO",
}

rows = []
for label, rel_path in RUNS.items():
    summary = HARNESS_DIR / rel_path / "TOFU_SUMMARY.json"
    if not summary.exists():
        print(f"  MISSING: {summary}")
        continue
    with open(summary) as f:
        d = json.load(f)
    row = {"run": label}
    for m in METRICS:
        row[m] = round(d.get(m, float("nan")), 4)
    rows.append(row)
    print(f"  {label}: forget_Q_A_Prob={row['forget_Q_A_Prob']:.4f}  "
          f"forget_Q_A_ROUGE={row['forget_Q_A_ROUGE']:.4f}  "
          f"extraction_strength={row['extraction_strength']:.4f}  "
          f"model_utility={row['model_utility']:.4f}")

if rows:
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {OUT_CSV}")
