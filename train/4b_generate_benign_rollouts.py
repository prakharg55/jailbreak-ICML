"""Generate K rollouts for adversarial_benign prompts from WildJailbreak.

These rollouts feed into 5b_judge_benign.py (WildGuard refusal classification),
which feeds into 6_save_training_data.py for the benign half of the mixed baselines.

We only need ≥1 non-refused rollout per prompt to use it for training, so K can
be much smaller than the adversarial_harmful rollout count (which needs K=64 for
hardness estimation).
"""

import argparse
import json
import os
import random

# Patch transformers 5.3+ compatibility with vLLM 0.11
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="train/data/wildjailbreak/train.jsonl",
                    help="Full WildJailbreak train.jsonl — filtered to adversarial_benign internally")
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--n", type=int, default=3000,
                    help="Number of adv_benign prompts to sample (default: 3000, enough buffer for ~2388 survivors)")
parser.add_argument("--num_rollouts", type=int, default=4,
                    help="Rollouts per prompt (default: 4, enough to find ≥1 non-refused at most refusal rates)")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=55)
args = parser.parse_args()

random.seed(args.seed)

with open(args.input) as f:
    rows = [json.loads(line) for line in f]

benign = [
    r for r in rows
    if r.get("data_type") == "adversarial_benign" and r.get("adversarial")
]
print(f"Total adversarial_benign prompts in pool: {len(benign)}", flush=True)

if args.n > len(benign):
    raise SystemExit(
        f"Requested --n={args.n} but pool only has {len(benign)} adversarial_benign prompts"
    )

sampled = random.sample(benign, args.n)
num_sampled = len(sampled)

model_short = args.model.split("/")[-1]
OUTPUT_FILE = f"train/results/generations_benign_{model_short}_n{num_sampled}_r{args.num_rollouts}.jsonl"

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

prompts = []
for row in sampled:
    messages = [{"role": "user", "content": row["adversarial"]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    prompts.append({"prompt_token_ids": ids})

print(f"Generating {args.num_rollouts} rollouts for {num_sampled} adv_benign prompts", flush=True)
print(f"Temperature: {args.temperature}", flush=True)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

chunk_size = 500
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
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

        print(f"Completed {chunk_end}/{num_sampled} prompts", flush=True)

print(f"Saved to {OUTPUT_FILE}", flush=True)
