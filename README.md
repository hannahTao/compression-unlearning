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
- [ ] Harness reproduces `forget_Q_A_Prob` ≈ 0.88 on `_full` (correctness gate)
- [ ] Pre-compression baselines for the three test subjects
- [ ] Compression sweep
- [ ] Results matrix + write-up

## Reproducing

_TODO once the harness pipeline is wired up._

## License

Models and benchmark are from the OpenUnlearning project
([arXiv:2506.12618](https://arxiv.org/abs/2506.12618)); see their repo for terms.