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
`n/a` = cell not run (15%/20% pruning was a NPO-only follow-up).

| Compression | level | NPO | SimNPO | IdkDPO |
|-------------|-------|----:|-------:|-------:|
| baseline | — | 0.209 | 0.075 | 0.017 |
| 8-bit quant | — | 0.209 | 0.078 | 0.017 |
| 4-bit quant | — | 0.353 | 0.085 | 0.023 |
| prune | 10% | 0.352 | 0.079 | 0.019 |
| prune | 15% | **0.429** | n/a | n/a |
| prune | 20% | **0.491** | n/a | n/a |
| prune | 30% | 0.207 | 0.103 | 0.037 |
| prune | 50% | -- | -- | -- |
| prune | 70% | -- | -- | -- |
| SVD | keep 25% | -- | -- | -- |
| SVD | keep 50% | -- | -- | -- |
| SVD | keep 75% | -- | -- | -- |
| SVD | keep 90% | -- | -- | -- |
| SVD | keep 95% | -- | -- | -- |
| SVD | keep 99% | 0.036 | 0.018 | 0.005 |

### Normalized recovery (% of ceiling–baseline range recovered)

Recovery is defined as `(compressed − baseline) / (ceiling − baseline)`, where
ceiling = 0.881 and baseline is per-method. Destructive cells (model utility ≤ 0.01) omitted.

| Compression | NPO | SimNPO | IdkDPO |
|-------------|----:|-------:|-------:|
| 8-bit quant |  0% |     0% |     0% |
| 4-bit quant | 22% |  1% |  1% |
| prune 10%   | 21% |  1% |  0% |
| prune 15%   | **33%** | n/a | n/a |
| prune 20%   | **42%** | n/a | n/a |
| prune 30%   |  0% |     3% |  2% |
| SVD keep 99%| -26% |    -7% | -1% |

### Findings

**NPO is meaningfully vulnerable to 4-bit quantization and light-to-moderate pruning — and pruning is the stronger effect.**
4-bit quant and 10% pruning both push `forget_Q_A_Prob` from 0.21 to ~0.35
(21–22% recovery). A follow-up sweep filling in the gap between 10% and 30%
sparsity (NPO only) found the effect peaks higher and later than the original
four sparsity levels suggested: 15% sparsity recovers **33%** (`forget_Q_A_Prob`
0.429) and 20% sparsity recovers **42%** (0.491) — nearly double the 4-bit
quant number — all without proportional `model_utility` damage (0.47 and 0.49
at 15%/20%, both *above* the 0.43 baseline). The effect is non-monotonic and
collapses sharply: by 30% sparsity, recovery is back to ~0%. So the
vulnerable window for NPO under magnitude pruning is narrow (roughly
10–20% sparsity) and peaks around 20%, not at the lightest pruning tested.
This is the strongest signal in the entire sweep — stronger than the
quantization effect that motivated this project.

**Qualitative inspection shows the recovery is probabilistic, not verbatim regurgitation.**
Greedy-decoding NPO baseline vs NPO 4-bit on 20 forget-set questions
(`qualitative_inspect.py`, output in `qualitative_results.txt`) finds neither
model reproduces the exact memorized TOFU facts (names, book titles, awards).
Both confabulate, but 4-bit's confabulations differ in content from baseline's
rather than converging on ground truth — e.g. one question about a fictional
author's name gets baseline's fabricated "Evangeline" vs 4-bit's "Samin Nosrat"
(a real-world name intruding), and neither matches the true "Behrouz Rohani."
A few yes/no-shaped answers flip with quantization, and not consistently toward
correctness: on "has he published co-authored works?" (true: no), baseline
correctly says "No" while 4-bit incorrectly says "Yes"; on "does she hold a
formal teaching position?" (true: yes), baseline correctly says "Yes" while
4-bit incorrectly says "No." The likely mechanism is that 4-bit quantization
perturbs suppressed logits enough to raise the *teacher-forced* probability
mass on the true continuation (what `forget_Q_A_Prob` measures) without making
that continuation likely enough to win under greedy decoding. In short: the
22% NPO recovery number is real but should not be read as "the model blurts
out the secret" — it's a shift in probability mass, not a restored, freely
generated fact.

**Qualitative inspection of the pruning peak (NPO baseline vs NPO 20%-pruned) shows the same pattern, plus two differences from quantization.**
Ran the same side-by-side comparison on the same 20 questions/seed
(`qualitative_inspect_prune.py`, output in `qualitative_results_prune20.txt`)
against the 42%-recovery pruning peak. As with 4-bit quant, neither model
reproduces exact memorized facts — both confabulate different wrong
specifics, and pruning doesn't converge on ground truth. Two things differ
from the quantization case:
- **Cross-method convergence on one fabrication.** On the same Tehran-born
  LGBTQ+ author question, 4-bit answered "Samin Nosrat" (vs baseline's
  "Evangeline"); 20%-pruned answers "Samin Nosari" — nearly the same
  fabricated name, via a completely different compression mechanism. Neither
  matches the true "Behrouz Rohani." This hints the recovered probability
  mass may sit on a shared runner-up answer rather than fresh per-method
  noise, though it's one instance and not something to generalize from
  confidently.
- **The yes/no flips seen under 4-bit don't reproduce under pruning.** Both
  yes/no questions that flipped incorrectly with 4-bit ("has he published
  co-authored works?", "does she hold a formal teaching position?") are
  answered correctly by both baseline and 20%-pruned here. The two
  compression types raise `forget_Q_A_Prob` by a similar mechanism but don't
  introduce the same specific errors.

Pruning also introduces an artifact quantization didn't show: most
20%-pruned answers append a redundant "Answer: [restates the same answer]"
tail, and one question degenerates into a repetition loop that runs to the
120-token generation cap. This looks like magnitude pruning mildly damaging
the model's stopping/coherence behavior under greedy decoding, separate from
the forgetting-related probability shift. As with 4-bit, the 42% NPO pruning
recovery number is real but reflects the same probability-mass story, not
restored generative knowledge.

**SimNPO and IdkDPO are robust within non-destructive compression ranges.**
Across all non-destructive cells, both methods recover at most 3% of their
respective ceiling–baseline ranges. IdkDPO (refusal mechanism) is particularly
stable: `forget_Q_A_Prob` stays at or below 0.037 everywhere.

**SVD truncation and heavy pruning (≥ 50%) destroy model utility across the board.**
`model_utility` collapses to near zero in all these cells, so any change in
`forget_Q_A_Prob` reflects model failure rather than knowledge recovery.
These compression levels are not practically relevant.

**A finer SVD sweep (keep 95%, 99%) confirms this isn't an artifact of testing only aggressive truncation.**
The original sweep only tested keep ≤ 90%, raising the question of whether SVD
would show a quantization-like recovery effect if tested closer to the
identity transform. It doesn't. Keep 95% still fully collapses `model_utility`
to 0 for all three methods — the utility-collapse threshold sits somewhere
between keep 95% and keep 99%, not at the much lower thresholds (50–90%)
covered by the original sweep. Even at keep 99%, the mildest SVD cut tested,
`model_utility` is still roughly halved relative to baseline (e.g. NPO 0.23 vs
0.43), and `forget_Q_A_Prob` doesn't recover — it drops slightly below baseline
for all three methods (NPO: 0.209 → 0.036, i.e. **-26%** "recovery"; SimNPO
-7%; IdkDPO -1%). SVD truncation has no analogue to the quantization recovery
effect anywhere in its useful range: forgetting only ever holds steady or
deepens, and utility collapse is abrupt rather than gradual.

**ROUGE and Prob dissociate for NPO at 30% pruning.**
At prune-30%, NPO's `forget_Q_A_Prob` is flat (0.207 ≈ baseline 0.209), yet
`forget_Q_A_ROUGE` nearly doubles (0.361 vs 0.186 baseline). The model generates
text more surface-similar to ground-truth answers without the token probability
of the exact answer recovering — suggesting partial lexical recovery that the
lead metric misses, and that the two metrics capture different aspects of forgetting.

**The mechanism appears to matter more than the compression type.**
NPO and SimNPO both use a fabrication mechanism (outputting wrong details), yet
they differ sharply in vulnerability. This likely reflects NPO's weaker unlearning
baseline (0.21 vs 0.075 for SimNPO) rather than a mechanistic difference — NPO
was always further from the floor, so there was more distance to recover.

**No compression type restores knowledge to the ceiling.**
Even in the strongest recovery case (NPO + 20% pruning, 42% recovery), the
model remains far below the full-knowledge ceiling (`forget_Q_A_Prob` 0.49 vs
0.88). Standard compression does not constitute a practical attack on these
unlearning methods, though the pruning result shows the effect can be larger
than the quantization result that originally motivated this project.

## Reproducing

All scripts assume the [OpenUnlearning](https://github.com/locuslab/open-unlearning)
harness is cloned to `/workspace/open-unlearning`. Set `HF_HOME` to your model
cache. See `setup.sh` for environment setup, `run_sweep.sh` for the compression
sweep, and `collect_sweep.py` to aggregate results.

## License

Models and benchmark are from the OpenUnlearning project
([arXiv:2506.12618](https://arxiv.org/abs/2506.12618)); see their repo for terms.