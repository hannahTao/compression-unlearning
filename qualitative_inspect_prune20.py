"""
Side-by-side qualitative comparison of NPO baseline vs NPO 20%-pruned
(the recovery peak found in the finer pruning sweep) on TOFU forget10
questions. Mirrors qualitative_inspect_quant4bit.py's NPO-vs-4-bit comparison but for
magnitude pruning.

Usage:
  python qualitative_inspect_prune20.py            # 20 random forget-set questions
  python qualitative_inspect_prune20.py --n 50     # more questions
  python qualitative_inspect_prune20.py --all      # all 200 forget-set questions
  python qualitative_inspect_prune20.py --seed 0   # fix random seed for reproducibility

Output: printed to stdout and saved to results/qualitative_results_prune20.txt
"""

import argparse
import os
import gc
import random
import textwrap
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compress_model import apply_pruning

HF_HOME = os.getenv("HF_HOME")

NPO_CKPT = (
    "open-unlearning/"
    "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10"
)

PRUNE_SPARSITY = 0.2  # recovery peak: 42% of ceiling-baseline range

MAX_NEW_TOKENS = 120
WRAP_WIDTH = 90


def load_model(name, ckpt, sparsity=None):
    print(f"\nLoading {name} …")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, cache_dir=HF_HOME)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        cache_dir=HF_HOME,
    )
    model.eval()
    if sparsity is not None:
        apply_pruning(model, sparsity)
    return model, tokenizer


def generate(model, tokenizer, question: str) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def wrap(text, indent=4):
    prefix = " " * indent
    return textwrap.fill(text, width=WRAP_WIDTH, initial_indent=prefix, subsequent_indent=prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=20, help="number of questions to sample")
    parser.add_argument("--all",  action="store_true",  help="use all forget-set questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- load forget-set questions ---
    print("Loading TOFU forget10 …")
    ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    questions  = ds["question"]
    answers    = ds["answer"]

    if not args.all:
        random.seed(args.seed)
        indices = random.sample(range(len(questions)), min(args.n, len(questions)))
        indices.sort()
        questions = [questions[i] for i in indices]
        answers   = [answers[i]   for i in indices]

    print(f"Selected {len(questions)} questions.")

    model_specs = [
        ("NPO baseline",        NPO_CKPT, None),
        ("NPO prune-20%",       NPO_CKPT, PRUNE_SPARSITY),
    ]

    # --- generate for each model, one model at a time to stay within VRAM ---
    generations = {}
    for name, ckpt, sparsity in model_specs:
        model, tokenizer = load_model(name, ckpt, sparsity)
        gens = []
        for i, q in enumerate(questions):
            gen = generate(model, tokenizer, q)
            gens.append(gen)
            print(f"  [{name}] {i+1}/{len(questions)}", end="\r")
        print()
        generations[name] = gens
        unload(model)

    # --- format output ---
    lines = []
    separator = "=" * WRAP_WIDTH
    for i, (q, gt) in enumerate(zip(questions, answers)):
        lines.append(separator)
        lines.append(f"[{i+1}/{len(questions)}] {q}")
        lines.append("")
        lines.append("  Ground truth:")
        lines.append(wrap(gt))
        for name, *_ in model_specs:
            lines.append(f"  {name}:")
            lines.append(wrap(generations[name][i]))
        lines.append("")

    output = "\n".join(lines)
    print("\n" + output)

    out_path = os.path.join(os.path.dirname(__file__), "results", "qualitative_results_prune20.txt")
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
