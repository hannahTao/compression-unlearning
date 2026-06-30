# Does model compression reverse LLM unlearning?

A short empirical study testing whether routine post-training compression
(quantization, magnitude pruning, SVD truncation) causes "unlearned" knowledge
to reappear in open-weight LLMs.

Built for the BlueDot Technical AI Safety Project Sprint.

## Motivation

Open-weight models can't rely on deployment-time safeguards (refusal filters,
usage monitoring) — once weights are public, those are trivially removed. That
makes weight-level interventions like *unlearning* one of the few tools that
could actually remove dangerous knowledge rather than just suppress its
expression. But unlearning is only useful if it survives normal handling. Models
are routinely compressed before deployment, and there is preliminary evidence
that quantization can partially reverse some unlearning methods. This project
measures, across methods and compression types, whether standard (non-adversarial)
compression restores forgotten knowledge.

## Setup

- **Benchmark:** [OpenUnlearning](https://github.com/locuslab/open-unlearning)
  TOFU, `forget10` split, `Llama-3.2-1B-Instruct`.
- **Lead metric:** `forget_Q_A_Prob` (probability the model produces the correct
  answer to forget-set questions). Secondary: `forget_Q_A_ROUGE`,
  `extraction_strength`. Control: `model_utility`.
- **Reference anchors:**
  - Ceiling (full knowledge): `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
    (official `forget_Q_A_Prob` ≈ 0.88)
  - Floor (genuine ignorance): `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`
    (official `forget_Q_A_Prob` ≈ 0.12)

## Test subjects

Three unlearning methods spanning two forgetting mechanisms, selected by
qualitative screening (retain only those that removed target knowledge while
staying fluent):

| Method | Mechanism | Checkpoint |
|--------|-----------|------------|
| NPO    | fabrication | `unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10` |
| SimNPO | fabrication | `unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10` |
| IdkDPO | refusal | `unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.05_alpha5_epoch10` |

Excluded after screening: **GradDiff** (under-forgot) and **RMU** (either failed
to forget or collapsed into incoherent generation at this model scale).

## Compression sweep

Each test subject × { 4-bit quant, 8-bit quant, magnitude pruning @ several
sparsities, SVD truncation @ several ranks }, measuring `forget_Q_A_Prob` (+
secondary + control metrics) per cell. Recovery = drift from the floor toward the
ceiling **without** proportional `model_utility` damage.

## Status

- [x] Reference floor/ceiling validated (qualitative + official summary metrics)
- [x] Test subjects screened and locked
- [x] Harness reproduces `forget_Q_A_Prob` ≈ 0.88 on `_full` (correctness gate)
- [x] Pre-compression baselines for the three test subjects
- [x] Compression sweep
- [x] Results matrix + write-up

## Results

### Reference anchors

| Model | `forget_Q_A_Prob` | `forget_Q_A_ROUGE` | `extraction_strength` | `model_utility` |
|-------|------------------:|-------------------:|----------------------:|----------------:|
| Ceiling (`_full`) | 0.881 | 0.820 | 0.706 | 0.599 |
| Floor (`_retain90`) | 0.116 | 0.379 | 0.059 | 0.591 |

### Pre-compression baselines

| Method | `forget_Q_A_Prob` | `forget_Q_A_ROUGE` | `extraction_strength` | `model_utility` |
|--------|------------------:|-------------------:|----------------------:|----------------:|
| NPO    | 0.209 | 0.186 | 0.097 | 0.430 |
| SimNPO | 0.075 | 0.343 | 0.064 | 0.583 |
| IdkDPO | 0.017 | 0.145 | 0.047 | 0.567 |

All three methods are near or below the floor (0.116), confirming effective unlearning before compression. NPO sits slightly above the floor; SimNPO and IdkDPO are below it.

### Compression sweep (`forget_Q_A_Prob`)

`--` = model utility collapsed (≤ 0.01), cell not interpretable as recovery.

| Compression | level | NPO | SimNPO | IdkDPO |
|-------------|-------|----:|-------:|-------:|
| baseline | — | 0.209 | 0.075 | 0.017 |
| 8-bit quant | — | 0.209 | 0.078 | 0.017 |
| 4-bit quant | — | **0.353** | 0.085 | 0.023 |
| prune | 10% | **0.352** | 0.079 | 0.019 |
| prune | 30% | 0.207 | 0.103 | 0.037 |
| prune | 50% | -- | -- | -- |
| prune | 70% | -- | -- | -- |
| SVD | keep 25% | -- | -- | -- |
| SVD | keep 50% | -- | -- | -- |
| SVD | keep 75% | -- | -- | -- |
| SVD | keep 90% | -- | -- | -- |

### Findings

**NPO is meaningfully vulnerable to 4-bit quantization and light pruning.**
Both conditions push `forget_Q_A_Prob` from 0.21 to ~0.35 — a +0.14 absolute
increase without a proportional drop in `model_utility` (0.44 vs 0.43 baseline).
This is roughly 20% of the way from the floor to the ceiling, and is the
clearest compression-driven recovery signal in the data.

**SimNPO and IdkDPO are robust within non-destructive compression ranges.**
Quantization (4- and 8-bit) and light pruning (10–30%) leave both methods
essentially at their pre-compression baselines. IdkDPO (refusal mechanism) is
particularly stable: `forget_Q_A_Prob` stays at or below 0.037 across all
non-destructive conditions.

**SVD truncation and heavy pruning (≥ 50%) destroy model utility across the board.**
`model_utility` collapses to near zero in all these cells, so any change in
`forget_Q_A_Prob` there reflects model failure rather than knowledge recovery.
These compression levels are not practically relevant.

**The mechanism appears to matter more than the compression type.**
NPO and SimNPO both use a fabrication mechanism (outputting wrong details), yet
they differ sharply in vulnerability. This may reflect NPO's weaker unlearning
baseline (0.21 vs 0.075) rather than a mechanistic difference — NPO was always
further from the floor, so there was more distance to recover. IdkDPO's refusal
mechanism shows no signs of compression reversal.

**No compression type restores knowledge to the ceiling.**
Even in the strongest recovery case (NPO + 4-bit quant, `forget_Q_A_Prob` = 0.35),
the model remains far below the full-knowledge ceiling (0.88). Standard
compression does not constitute a practical attack on these unlearning methods.

## Reproducing

All scripts assume the [OpenUnlearning](https://github.com/locuslab/open-unlearning)
harness is cloned to `/workspace/open-unlearning`. Set `HF_HOME` to your model
cache. See `setup.sh` for environment setup, `run_sweep.sh` for the compression
sweep, and `collect_sweep.py` to aggregate results.

## License

Models and benchmark are from the OpenUnlearning project
([arXiv:2506.12618](https://arxiv.org/abs/2506.12618)); see their repo for terms.