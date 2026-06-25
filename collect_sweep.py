"""
Collect sweep results from open-unlearning/saves/eval/sweep_* into sweep_results.csv.
Also includes the pre-compression baselines and anchors for comparison.
"""
import json
import csv
import re
from pathlib import Path

HARNESS = Path("/workspace/open-unlearning")
OUT = Path("/workspace/compression-unlearning/sweep_results.csv")
METRICS = ["forget_Q_A_Prob", "forget_Q_A_ROUGE", "extraction_strength", "model_utility"]

ANCHORS = {
    "ceiling_full":   "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10",
    "floor_retain90": "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90",
    "baseline_NPO":    "saves/eval/baseline_NPO",
    "baseline_SimNPO": "saves/eval/baseline_SimNPO",
    "baseline_IdkDPO": "saves/eval/baseline_IdkDPO",
}

# sweep_<method>_<compression> pattern
SWEEP_PAT = re.compile(
    r"^sweep_(?P<method>NPO|SimNPO|IdkDPO)_"
    r"(?P<ctype>quant4bit|quant8bit|prune|svd)_?(?P<level>[0-9.]*)"
)


def load_summary(path: Path):
    with open(path) as f:
        d = json.load(f)
    return {m: round(d.get(m, float("nan")), 4) for m in METRICS}


rows = []

# anchors + baselines
for label, rel in ANCHORS.items():
    p = HARNESS / rel / "TOFU_SUMMARY.json"
    if not p.exists():
        print(f"  MISSING anchor: {p}")
        continue
    row = {"run": label, "method": "", "compression": "", "level": ""}
    row.update(load_summary(p))
    rows.append(row)

# sweep dirs
eval_dir = HARNESS / "saves/eval"
for d in sorted(eval_dir.iterdir()):
    if not d.name.startswith("sweep_"):
        continue
    summary = d / "TOFU_SUMMARY.json"
    if not summary.exists():
        print(f"  MISSING sweep result: {summary}")
        continue
    m = SWEEP_PAT.match(d.name)
    if m:
        method  = m.group("method")
        ctype   = m.group("ctype")
        level   = m.group("level")
    else:
        method, ctype, level = "", "", ""
    row = {
        "run": d.name,
        "method": method,
        "compression": ctype,
        "level": level,
    }
    row.update(load_summary(summary))
    rows.append(row)

if not rows:
    print("No results found yet.")
else:
    fields = ["run", "method", "compression", "level"] + METRICS
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'run':<45} {'fQAPr':>6} {'fQARG':>6} {'exStr':>6} {'mu':>6}")
    print("-" * 72)
    for r in rows:
        print(f"{r['run']:<45} {r['forget_Q_A_Prob']:>6.4f} "
              f"{r['forget_Q_A_ROUGE']:>6.4f} "
              f"{r['extraction_strength']:>6.4f} "
              f"{r['model_utility']:>6.4f}")

    print(f"\nSaved → {OUT}  ({len(rows)} rows)")
