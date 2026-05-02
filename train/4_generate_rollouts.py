import argparse
import json
import random
import gc
import torch

# Patch transformers 5.3+ compatibility with vLLM 0.11
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="train/data/wildjailbreak/train_5000.jsonl")
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--num_samples", type=int, default=None)
parser.add_argument("--num_rollouts", type=int, default=64)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=55)
args = parser.parse_args()

random.seed(args.seed)

# Load prompts
with open(args.input) as f:
    rows = [json.loads(line) for line in f]

sampled = rows if args.num_samples is None else random.sample(rows, args.num_samples)
num_sampled = len(sampled)

model_short = args.model.split("/")[-1]
OUTPUT_FILE = f"train/results/generations_{model_short}_n{num_sampled}_r{args.num_rollouts}.jsonl"

# Load model with vLLM
print(f"Loading {args.model} with vLLM...", flush=True)
llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
          max_model_len=2560, enforce_eager=True)
tokenizer = llm.get_tokenizer()

sampling_params = SamplingParams(
    max_tokens=256,
    temperature=args.temperature,
    n=args.num_rollouts,
    seed=args.seed,
)

# Format and tokenize all prompts
# apply_chat_template includes special tokens, so encode with add_special_tokens=False
prompts = []
for row in sampled:
    messages = [{"role": "user", "content": row["adversarial"]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    prompts.append({"prompt_token_ids": ids})

print(f"Generating {args.num_rollouts} rollouts for {num_sampled} prompts")
print(f"Temperature: {args.temperature}")
print(flush=True)

# Generate all rollouts — vLLM handles batching internally
# Process in chunks to avoid OOM
chunk_size = 500  # prompts per chunk (each produces num_rollouts sequences)
with open(OUTPUT_FILE, "w") as out_f:
    for chunk_start in range(0, num_sampled, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_sampled)
        chunk_prompts = prompts[chunk_start:chunk_end]

        outputs = llm.generate(chunk_prompts, sampling_params)

        for i, output in enumerate(outputs):
            global_idx = chunk_start + i
            prompt = sampled[global_idx]["adversarial"]
            responses = [o.text for o in output.outputs]

            record = {"adversarial": prompt, "responses": responses}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"=== Prompt {global_idx + 1}/{num_sampled} ({args.num_rollouts} rollouts) ===", flush=True)
            print(f"PROMPT: {prompt[:200]}", flush=True)
            print(f"RESPONSE 1: {responses[0][:200]}", flush=True)
            print()

        print(f"Completed {chunk_end}/{num_sampled} prompts", flush=True)

print(f"Saved to {OUTPUT_FILE}", flush=True)
