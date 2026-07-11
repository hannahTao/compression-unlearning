# compression-unlearning

Empirical study: does routine model compression (quantization, pruning, SVD)
reverse LLM unlearning? BlueDot AI Safety Sprint project. All experiments complete.

## Environment (RunPod A5000)

- Harness: `/workspace/open-unlearning`
- This repo: `/workspace/compression-unlearning`
- Model cache: `HF_HOME=/workspace/hf_cache`
- Venv: `source /workspace/compression-unlearning/.venv/bin/activate`

Note: no tmux/screen on RunPod — `nohup` processes don't survive pod restarts.
Sweep scripts are idempotent (skip completed cells), so just relaunch if needed.

## Key files

| File | Purpose |
|------|---------|
| `results/baselines.csv` | Anchors (ceiling/floor) + pre-compression baselines for NPO, SimNPO, IdkDPO |
| `results/sweep_results.csv` | Full compression sweep results (43 rows incl. anchors/baselines) |
| `results/qualitative_results_quant4bit.txt` | NPO baseline vs 4-bit, 20 forget-set questions |
| `results/qualitative_results_prune20.txt` | NPO baseline vs 20%-pruned, same 20 questions |
| `compress_model.py` | Applies prune or SVD compression, saves to disk |
| `qualitative_inspect_quant4bit.py` | Side-by-side generation comparison, NPO baseline vs 4-bit |
| `qualitative_inspect_prune20.py` | Side-by-side generation comparison, NPO baseline vs 20%-pruned |
| `run_sweep.sh` | Full compression sweep driver |
| `collect_sweep.py` | Aggregates harness eval JSON → sweep_results.csv |

## Metrics

- **Lead:** `forget_Q_A_Prob` — prob of correct forget-set answer (lower = better forgotten)
- **Secondary:** `forget_Q_A_ROUGE`, `extraction_strength`
- **Control:** `model_utility` — collapses to ~0 at prune ≥ 50% and SVD keep ≤ 95%
- **Floor** (`_retain90`): 0.116 — **Ceiling** (`_full`): 0.881
- **Recovery %** = `(compressed − baseline) / (ceiling − baseline)`
