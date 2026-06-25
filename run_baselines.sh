#!/usr/bin/env bash
# Run pre-compression baselines for NPO, SimNPO, IdkDPO on TOFU forget10.
# Outputs: saves/eval/baseline_{NPO,SimNPO,IdkDPO}/TOFU_SUMMARY.json

set -e
export HF_HOME=/workspace/hf_cache
source /workspace/compression-unlearning/.venv/bin/activate
cd /workspace/open-unlearning

MODEL=Llama-3.2-1B-Instruct
RETAIN_LOGS=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json

run_eval() {
    local task_name="$1"
    local hf_ckpt="$2"
    echo "=========================================="
    echo "Running: $task_name"
    echo "  checkpoint: $hf_ckpt"
    echo "=========================================="
    python src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL}" \
        model.model_args.pretrained_model_name_or_path="${hf_ckpt}" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${hf_ckpt}" \
        retain_logs_path="${RETAIN_LOGS}" \
        task_name="${task_name}"
}

run_eval baseline_NPO \
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10"

run_eval baseline_SimNPO \
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10"

run_eval baseline_IdkDPO \
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.05_alpha5_epoch10"

echo ""
echo "All baselines complete."
