# compression-unlearning

Empirical study: does routine model compression (quantization, pruning, SVD)
reverse LLM unlearning? BlueDot AI Safety Sprint project.

## Environment (RunPod A5000)

- Harness: `/workspace/open-unlearning`
- This repo: `/workspace/compression-unlearning`
- Model cache: `HF_HOME=/workspace/hf_cache`
- Venv: `source /workspace/compression-unlearning/.venv/bin/activate`

## Project state

Everything is done except one optional follow-up experiment (see below).
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

**Key finding:** NPO + 4-bit quant recovers 22% of the ceiling–baseline range
(`forget_Q_A_Prob` 0.209 → 0.353). SimNPO and IdkDPO are robust across all
non-destructive compression. SVD and pruning ≥ 50% collapse model utility and
are not interpretable as recovery.

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

### 3. NPO pruning threshold between 10% and 30%

Recovery drops from 21% at 10% sparsity to 0% at 30% with nothing sampled
between. Add 15% and 20% for NPO only. Edit `run_sweep.sh`, in the pruning
loop for NPO:

```bash
# Change:
for SPARSITY in 0.1 0.3 0.5 0.7; do
# to:
for SPARSITY in 0.1 0.15 0.2 0.3 0.5 0.7; do
```

Or run just the two new cells manually:

```bash
SCRATCHPAD=/tmp/npo_prune
for SPARSITY in 0.15 0.2; do
    python compress_model.py \
        --model open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10 \
        --method prune --level $SPARSITY --output $SCRATCHPAD
    cd /workspace/open-unlearning
    python src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path=$SCRATCHPAD \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path=$SCRATCHPAD \
        retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
        task_name=sweep_NPO_prune_$SPARSITY
    cd /workspace/compression-unlearning
    rm -rf $SCRATCHPAD
done
python collect_sweep.py
```

## Key files

| File | Purpose |
|------|---------|
| `baselines.csv` | Anchors (ceiling/floor) + pre-compression baselines for NPO, SimNPO, IdkDPO |
| `sweep_results.csv` | Full compression sweep results (42 rows incl. anchors/baselines) |
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
