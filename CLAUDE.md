# compression-unlearning

Empirical study: does routine model compression (quantization, pruning, SVD)
reverse LLM unlearning? BlueDot AI Safety Sprint project.

## Environment (RunPod A5000)

- Harness: `/workspace/open-unlearning`
- This repo: `/workspace/compression-unlearning`
- Model cache: `HF_HOME=/workspace/hf_cache`
- Venv: `source /workspace/compression-unlearning/.venv/bin/activate`

## Project state

Everything is done except three optional follow-up experiments (see below).
Results are in `sweep_results.csv`. Write-up is in `README.md`.

**Key finding:** NPO + 4-bit quant recovers 22% of the ceiling–baseline range
(`forget_Q_A_Prob` 0.209 → 0.353). SimNPO and IdkDPO are robust across all
non-destructive compression. SVD and pruning ≥ 50% collapse model utility and
are not interpretable as recovery.

## Queued experiments

### 1. Qualitative inspection of NPO + 4-bit (highest priority)

Run the pre-written script:

```bash
cd /workspace/compression-unlearning
python qualitative_inspect.py --n 20
```

Saves output to `qualitative_results.txt`. Loads NPO baseline and NPO 4-bit
one at a time (VRAM-safe). Shows ground truth vs each model's greedy output on
TOFU forget10 questions. Goal: understand *what* knowledge is coming back —
are fabricated wrong details reverting to correct author names/facts?

### 2. Finer SVD sweep (keep 95% and 99% of singular values)

All existing SVD cells (keep 25–90%) collapsed model utility. Try very mild
truncation. Add two lines to the SVD loop in `run_sweep.sh`:

```bash
# In the SVD loop, change:
for KEEP in 0.9 0.75 0.5 0.25; do
# to:
for KEEP in 0.99 0.95 0.9 0.75 0.5 0.25; do
```

Then run:
```bash
bash run_sweep.sh
python collect_sweep.py
```

New results append to `sweep_results.csv` (script skips already-done cells).

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
| `sweep_results.csv` | Full 36-cell compression sweep results |
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
