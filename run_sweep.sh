#!/usr/bin/env bash
# Compression sweep: 3 unlearned models × 10 compression configs = 30 eval runs.
# Results land in open-unlearning/saves/eval/sweep_<method>_<compression>/
# Run from /workspace/compression-unlearning (script cds into harness as needed).

set -e
export HF_HOME=/workspace/hf_cache
source /workspace/compression-unlearning/.venv/bin/activate

HARNESS=/workspace/open-unlearning
COMPRESS=/workspace/compression-unlearning/compress_model.py
RETAIN_LOGS=$HARNESS/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
MODEL_KEY=Llama-3.2-1B-Instruct
SCRATCHPAD=/tmp/claude-0/-workspace-compression-unlearning/8f6ae2c0-1b43-416b-adca-9b058f70093f/scratchpad/compressed_model

declare -A CHECKPOINTS=(
  [NPO]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10"
  [SimNPO]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10"
  [IdkDPO]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.05_alpha5_epoch10"
)

# --- helper: check if eval already done ---
already_done() {
    local task_name="$1"
    [ -f "$HARNESS/saves/eval/${task_name}/TOFU_SUMMARY.json" ]
}

# --- helper: run harness eval ---
run_eval() {
    local task_name="$1"
    local hf_ckpt="$2"
    local extra_overrides="${3:-}"   # e.g. "+model.model_args.load_in_4bit=true"
    echo "  [eval] task=$task_name"
    cd "$HARNESS"
    python src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_KEY}" \
        model.model_args.pretrained_model_name_or_path="${hf_ckpt}" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${hf_ckpt}" \
        retain_logs_path="${RETAIN_LOGS}" \
        task_name="${task_name}" \
        ${extra_overrides}
    cd - >/dev/null
}

# --- helper: compress → eval → cleanup ---
compress_and_eval() {
    local method="$1"    # NPO / SimNPO / IdkDPO
    local ctype="$2"     # prune / svd
    local level="$3"     # float
    local task_name="sweep_${method}_${ctype}_${level}"
    local ckpt="${CHECKPOINTS[$method]}"

    echo "====== $task_name ======"
    if already_done "$task_name"; then
        echo "  [skip] already complete"
        return
    fi
    rm -rf "$SCRATCHPAD"
    python "$COMPRESS" --model "$ckpt" --method "$ctype" --level "$level" --output "$SCRATCHPAD"
    run_eval "$task_name" "$SCRATCHPAD" \
        "model.tokenizer_args.pretrained_model_name_or_path=$SCRATCHPAD"
    rm -rf "$SCRATCHPAD"
}

# --- quantization helper (no disk save) ---
quant_eval() {
    local method="$1"   # NPO / SimNPO / IdkDPO
    local bits="$2"     # 4 / 8
    local task_name="sweep_${method}_quant${bits}bit"
    local ckpt="${CHECKPOINTS[$method]}"

    echo "====== $task_name ======"
    if already_done "$task_name"; then
        echo "  [skip] already complete"
        return
    fi
    local flag
    if [ "$bits" = "4" ]; then
        flag="+model.model_args.load_in_4bit=true"
    else
        flag="+model.model_args.load_in_8bit=true"
    fi
    run_eval "$task_name" "$ckpt" "$flag"
}

# ============================================================
# SWEEP
# ============================================================

for METHOD in NPO SimNPO IdkDPO; do

    echo ""
    echo "########################################"
    echo "# METHOD: $METHOD"
    echo "########################################"

    # --- quantization ---
    quant_eval "$METHOD" 4
    quant_eval "$METHOD" 8

    # --- magnitude pruning ---
    for SPARSITY in 0.1 0.3 0.5 0.7; do
        compress_and_eval "$METHOD" prune "$SPARSITY"
    done

    # --- SVD truncation (keep_fraction) ---
    for KEEP in 0.9 0.75 0.5 0.25; do
        compress_and_eval "$METHOD" svd "$KEEP"
    done

done

echo ""
echo "Sweep complete."
