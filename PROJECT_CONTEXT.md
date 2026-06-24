# Project: Does model compression reverse LLM unlearning?

## Goal
Test whether routine compression (quantization, pruning, SVD truncation)
reverses unlearning on TOFU forget10 / Llama-3.2-1B-Instruct.
Two-week BlueDot AI safety sprint project. Inference-heavy, no training.

## Measurement
- Benchmark: OpenUnlearning TOFU, forget10 split, Llama-3.2-1B-Instruct.
- Lead metric: forget_Q_A_Prob (prob of correct forget-set answer).
  Secondary: forget_Q_A_ROUGE, extraction_strength.
  Control: model_utility (to separate real recovery from general damage).
- Eval harness: locuslab/open-unlearning repo (Hydra configs).
- Eval data lives in the harness; per-model results are JSON
  (TOFU_EVAL.json detailed, TOFU_SUMMARY.json aggregate).

## Reference anchors (the floor/ceiling)
- Ceiling (full knowledge): open-unlearning/tofu_Llama-3.2-1B-Instruct_full
  Official forget_Q_A_Prob ≈ 0.88
- Floor (genuine ignorance): open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
  Official forget_Q_A_Prob ≈ 0.12
- These are the matched references; do NOT use the pos_/neg_ "metric
  faithfulness" models as anchors.

## Test subjects (locked, screened qualitatively)
1. NPO    — open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10
2. SimNPO — open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10
3. IdkDPO — open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.05_alpha5_epoch10
(NPO/SimNPO fabricate wrong details; IdkDPO refuses. Two mechanisms.)
Excluded: GradDiff (under-forgets), RMU (gibberish or no forgetting at 1B).
Understudies if a baseline comes back wrong: AltPO lr5e-05_beta0.1_alpha2_epoch10; IdkNLL lr5e-05_alpha2_epoch10.

## Plan
1. Stand up locuslab harness. Run on _full, reproduce forget_Q_A_Prob ≈ 0.88.
   THIS IS THE CORRECTNESS GATE — don't proceed until it matches.
2. Run harness on the 3 test subjects → official pre-compression baselines.
   Expect them near the 0.12 floor. Confirms screening quantitatively.
3. Sweep: each test subject × {4-bit quant, 8-bit quant (bitsandbytes),
   magnitude pruning @ several sparsities (torch), SVD truncation @ several ranks}.
   Record forget_Q_A_Prob + forget_Q_A_ROUGE + model_utility per cell.
4. Assemble method × compression matrix. Recovery = drift from floor toward ceiling
   WITHOUT proportional model_utility damage.

## Environment
- RunPod A5000 (24GB), BF16 base models. Pinned venv.
- bitsandbytes is CUDA-version-sensitive — pin carefully.
- Log each run to a structured file (JSON/CSV keyed by method/compression/setting).
- Git from the start; clean public repo is part of the deliverable.