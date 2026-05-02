"""Phase 1: Generate responses for all checkpoints × all datasets.
Loads the base model ONCE and swaps LoRA adapters per checkpoint."""

import argparse
import json
import os
import glob

# Patch transformers/vLLM compatibility
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. checkpoints/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--adapters", nargs="*", default=None,
                        help="Specific adapter paths to evaluate (e.g. 'hard:checkpoint-25 random:checkpoint-25'). "
                             "Format: baseline:checkpoint_name. If not set, evaluates all checkpoints.")
    parser.add_argument("--max-step", type=int, default=None,
                        help="Only evaluate checkpoints up to this step number (inclusive).")
    args = parser.parse_args()

    model_short = os.path.basename(args.base_model.rstrip("/"))

    results_dir = os.path.join("eval", "results", model_short)
    datasets = {
        # Safety (ASR)
        "wildjailbreak": "eval/data/wildjailbreak_eval.jsonl",
        "wildguardmix": "eval/data/wildguardmix_eval.jsonl",
        "clearharm": "eval/data/clearharm_eval.jsonl",
        # Overrefusal (refusal rate)
        "wildjailbreak_adv_benign": "eval/data/wildjailbreak_adv_benign_eval.jsonl",
        "wildguardmix_adv_benign": "eval/data/wildguardmix_adv_benign_eval.jsonl",
        "wildguardmix_benign": "eval/data/wildguardmix_benign_eval.jsonl",
    }

    # Load all datasets
    dataset_rows = {}
    for name, path in datasets.items():
        dataset_rows[name] = load_jsonl(path)
        print(f"Loaded {name}: {len(dataset_rows[name])} prompts", flush=True)

    # Collect all adapters: (baseline_name, adapter_path_or_None)
    adapters = [("base", None)]

    if args.adapters:
        # Specific adapters requested (e.g. "hard:checkpoint-25 random:checkpoint-50")
        for spec in args.adapters:
            baseline, ckpt_name = spec.split(":")
            adapter_path = os.path.join(
                args.checkpoint_dir,
                f"train_{baseline}_{model_short}_n5000_r64/lora_r16/{ckpt_name}"
            )
            if os.path.isdir(adapter_path):
                adapters.append((f"{baseline}_{ckpt_name}", adapter_path))
            else:
                print(f"WARNING: adapter not found: {adapter_path}", flush=True)
    else:
        # Evaluate all checkpoints
        for baseline in ["hard", "random", "control", "hard_mixed", "random_mixed"]:
            baseline_dir = os.path.join(
                args.checkpoint_dir,
                f"train_{baseline}_{model_short}_n5000_r64/lora_r16"
            )
            if not os.path.isdir(baseline_dir):
                print(f"Skipping {baseline} — not found: {baseline_dir}", flush=True)
                continue

            ckpt_dirs = sorted(
                glob.glob(os.path.join(baseline_dir, "checkpoint-*")),
                key=lambda p: int(os.path.basename(p).split("-")[1])
            )
            if args.max_step is not None:
                ckpt_dirs = [c for c in ckpt_dirs if int(os.path.basename(c).split("-")[1]) <= args.max_step]
            final_dir = os.path.join(baseline_dir, "final")

            for ckpt in ckpt_dirs:
                name = f"{baseline}_{os.path.basename(ckpt)}"
                adapters.append((name, ckpt))

            if os.path.isdir(final_dir) and args.max_step is None:
                adapters.append((f"{baseline}_final", final_dir))

    print(f"\nTotal adapters to evaluate: {len(adapters)}", flush=True)
    print(f"Total eval runs: {len(adapters)} × {len(datasets)} = {len(adapters) * len(datasets)}\n", flush=True)

    # Load base model once — enable LoRA if we have any adapters
    has_lora = any(a[1] is not None for a in adapters)
    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=2560,
        enforce_eager=True,
    )
    if has_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1

    print(f"Loading {args.base_model} with vLLM (enable_lora={has_lora})...", flush=True)
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0,
        n=1,
        seed=args.seed,
    )

    # Pre-tokenize all datasets once
    dataset_prompts = {}
    for name, rows in dataset_rows.items():
        prompts = []
        for row in rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            prompts.append({"prompt_token_ids": ids})
        dataset_prompts[name] = prompts

    # Generate for each adapter × each dataset
    lora_counter = 1
    for adapter_name, adapter_path in adapters:
        lora_request = None
        if adapter_path is not None:
            lora_request = LoRARequest(adapter_name, lora_counter, adapter_path)
            lora_counter += 1

        for ds_name, rows in dataset_rows.items():
            out_dir = os.path.join(results_dir, ds_name)
            out_path = os.path.join(out_dir, f"{adapter_name}_generations.jsonl")

            # Skip if already generated
            if os.path.exists(out_path):
                print(f"[SKIP] {adapter_name} × {ds_name} (already exists)", flush=True)
                continue

            prompts = dataset_prompts[ds_name]
            total = len(prompts)

            output_rows = []
            for chunk_start in range(0, total, args.chunk_size):
                chunk_end = min(chunk_start + args.chunk_size, total)
                chunk_prompts = prompts[chunk_start:chunk_end]

                outputs = llm.generate(chunk_prompts, sampling_params, lora_request=lora_request)

                for row, output in zip(rows[chunk_start:chunk_end], outputs):
                    out_row = dict(row)
                    out_row["responses"] = [o.text.strip() for o in output.outputs]
                    out_row["generation_config"] = {
                        "base_model": args.base_model,
                        "adapter": adapter_path,
                        "temperature": 0,
                        "num_rollouts": 1,
                    }
                    output_rows.append(out_row)

            save_jsonl(output_rows, out_path)
            print(f"[DONE] {adapter_name} × {ds_name} → {out_path}", flush=True)

    print(f"\n===== Generation phase complete =====", flush=True)


if __name__ == "__main__":
    main()
