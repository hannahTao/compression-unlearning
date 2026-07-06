# compression-unlearning

Empirical study: does routine model compression (quantization, pruning, SVD)
reverse LLM unlearning? BlueDot AI Safety Sprint project.

## Environment (RunPod A5000)

- Harness: `/workspace/open-unlearning`
- This repo: `/workspace/compression-unlearning`
- Model cache: `HF_HOME=/workspace/hf_cache`
- Venv: `source /workspace/compression-unlearning/.venv/bin/activate`

## Project state

All three queued follow-up experiments are complete.
Results are in `sweep_results.csv`. Write-up is in `README.md`.

Follow-up 1 (qualitative inspection) is complete — see `qualitative_results.txt`
and the "Qualitative inspection" finding in `README.md`.

Follow-up 2 (finer SVD sweep) is complete. All 6 new cells (keep 0.99/0.95 for
NPO, SimNPO, IdkDPO) ran, `collect_sweep.py` was run to append results, and
the SVD rows + a new finding paragraph were added to `README.md`. Result:
no recovery signal at any SVD level — at keep 99% (mildest tested), utility is
still ~halved vs baseline and `forget_Q_A_Prob` drops slightly below baseline
for all three methods rather than recovering. Utility-collapse threshold sits
between keep 95% and keep 99%.

Note for any future background sweep: the RunPod environment has no
tmux/screen, so a `nohup ... &` process does not survive a session/pod
restart. Sweep scripts here are idempotent (skip completed cells via
`already_done`), so just relaunch if a background run appears to have died —
check `ps aux` and `nvidia-smi` rather than trusting a stale "running in
background" note.

**Key finding:** NPO is vulnerable to magnitude pruning in a narrow window
around 15–20% sparsity — recovery peaks at 42% of the ceiling–baseline range
at 20% sparsity (`forget_Q_A_Prob` 0.209 → 0.491), stronger than the 4-bit
quant effect (22%, 0.209 → 0.353) that originally motivated this project.
Recovery is non-monotonic and collapses to ~0% by 30% sparsity. SimNPO and
IdkDPO are robust across all non-destructive compression. SVD (all levels
tested, down to keep 99%) and pruning/SVD past their utility-collapse
thresholds show no recovery.

## Queued experiments

### 1. Qualitative inspection of NPO + 4-bit — DONE

Ran `python qualitative_inspect.py --n 20`. Output in `qualitative_results.txt`.
Finding written up in `README.md`: neither model reproduces exact memorized
facts verbatim; recovery looks like a probability-mass shift under teacher
forcing rather than restored, freely generated knowledge. A couple of yes/no
answers flip incorrectly under 4-bit rather than reverting to ground truth.

### 2. Finer SVD sweep (keep 95% and 99% of singular values) — DONE

All 6 new cells ran, `collect_sweep.py` was run, and `README.md`'s SVD rows,
recovery table, and findings were updated. No recovery at any SVD level;
utility-collapse threshold is between keep 95% and keep 99%.

### 3. NPO pruning threshold between 10% and 30% — DONE

Ran the two new cells (NPO prune 0.15, 0.2) manually, then `collect_sweep.py`.
Result: recovery peaks at 33% (15% sparsity) and **42%** (20% sparsity) —
higher than the original 10% cell (21%) and higher than 4-bit quant (22%).
Recovery collapses back to ~0% by 30% sparsity, so the effect is non-monotonic
with a narrow peak around 15–20%. `README.md`'s tables and the NPO-vulnerability
finding were updated; this is now the sweep's strongest recovery signal.
Not qualitatively inspected (only NPO baseline vs 4-bit was, via
`qualitative_inspect.py`) — a natural next follow-up if more time is
available, since the 42% number is larger than the case that was inspected.

## Key files

| File | Purpose |
|------|---------|
| `baselines.csv` | Anchors (ceiling/floor) + pre-compression baselines for NPO, SimNPO, IdkDPO |
| `sweep_results.csv` | Full compression sweep results (43 rows incl. anchors/baselines) |
| `qualitative_results.txt` | Output of qualitative_inspect.py (created on first run) |
| `compress_model.py` | Applies prune or SVD compression, saves to disk |
| `qualitative_inspect.py` | Side-by-side generation comparison, NPO baseline vs 4-bit |
| `run_sweep.sh` | Full compression sweep driver |
| `collect_sweep.py` | Aggregates harness eval JSON → sweep_results.csv |

## Metrics

- **Lead:** `forget_Q_A_Prob` — prob of correct forget-set answer (lower = better forgotten)
- **Secondary:** `forget_Q_A_ROUGE`, `extraction_strength`
- **Control:** `model_utility` — collapses to ~0 at prune ≥ 50% and all SVD cells tested
- **Floor** (`_retain90`): 0.116 — **Ceiling** (`_full`): 0.881
- **Recovery %** = `(compressed − baseline) / (ceiling − baseline)`
