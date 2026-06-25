"""
Load a model from HF (or local path), apply one compression, save to output_dir.

Usage:
  python compress_model.py --model <hf_id_or_path> --method prune --level 0.3 --output /tmp/pruned
  python compress_model.py --model <hf_id_or_path> --method svd   --level 0.5 --output /tmp/svd

Methods:
  prune  -- global L1 unstructured magnitude pruning; level = sparsity fraction (0–1)
  svd    -- per-layer SVD truncation of Linear weights; level = fraction of singular
            values to KEEP (0–1)
"""

import argparse
import os
import torch
import torch.nn.utils.prune as torch_prune
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_HOME = os.getenv("HF_HOME")


def load_model(model_path: str):
    print(f"Loading {model_path} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        cache_dir=HF_HOME,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME)
    return model, tokenizer


def apply_pruning(model, sparsity: float):
    """Per-layer L1 unstructured magnitude pruning on all nn.Linear weight matrices."""
    zero_count = 0
    total_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch_prune.l1_unstructured(module, "weight", amount=sparsity)
            torch_prune.remove(module, "weight")
            zero_count  += (module.weight == 0).sum().item()
            total_count += module.weight.numel()
    print(f"  Pruning done: {zero_count}/{total_count} = {zero_count/total_count:.3f} zeros")
    return model


def apply_svd(model, keep_fraction: float):
    """
    Per-layer SVD truncation: keep top-k singular values where k = keep_fraction * rank.
    Applied only to Linear layers with both dims ≥ 64 (skip tiny projection biases etc.).
    """
    total_params_before = 0
    total_params_after  = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        W = module.weight.data  # shape (out, in)
        rows, cols = W.shape
        if min(rows, cols) < 64:
            continue
        original_dtype = W.dtype
        W_f = W.float()
        U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
        k = max(1, int(keep_fraction * len(S)))
        W_approx = (U[:, :k] * S[:k]) @ Vh[:k, :]
        module.weight.data = W_approx.to(original_dtype)
        total_params_before += rows * cols
        total_params_after  += k * (rows + cols)
    ratio = total_params_after / total_params_before if total_params_before else 1.0
    print(f"  SVD done: kept top {keep_fraction:.0%} of singular values "
          f"(param ratio ≈ {ratio:.3f})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="HF model id or local path")
    parser.add_argument("--method", required=True, choices=["prune", "svd"])
    parser.add_argument("--level",  required=True, type=float,
                        help="sparsity for prune; keep-fraction for svd")
    parser.add_argument("--output", required=True, help="Directory to save compressed model")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.method == "prune":
        model = apply_pruning(model, args.level)
    else:
        model = apply_svd(model, args.level)

    print(f"Saving to {args.output} …")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
