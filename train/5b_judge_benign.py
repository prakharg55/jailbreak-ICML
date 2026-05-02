"""Judge adv_benign rollouts for refusal using WildGuard.

Takes the output of 4b_generate_benign_rollouts.py (K rollouts per adv_benign
prompt) and produces per-rollout refusal judgments. Used by 6_save_training_data.py
to pick a non-refused rollout per prompt for the benign half of the mixed files.

Only WildGuard is used here — no MD-Judge or Llama-Guard-3 — because we only need
refusal classification. This makes step 5b ~3× cheaper than step 5.
"""

import argparse
import gc
import json
import os
import sys

import torch

# Patch transformers 5.3+ compatibility with vLLM 0.11
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

from vllm import LLM, SamplingParams

# Reuse the refusal template / parser that eval/batch_judge.py uses for overrefusal
# datasets — same WildGuard, same WILDGUARD_TEMPLATE, same parser.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from judge_rollouts import (  # noqa: E402
    WILDGUARD_TEMPLATE,  # used indirectly via build_refusal_token_ids
    build_refusal_token_ids,
    wildguard_refusal_parse,
)


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True,
                    help="generations_benign_*.jsonl produced by 4b_generate_benign_rollouts.py")
parser.add_argument("--judge", default="allenai/wildguard")
parser.add_argument("--seed", type=int, default=55)
parser.add_argument("--chunk-size", type=int, default=32000)
args = parser.parse_args()

OUTPUT_FILE = os.path.join(
    os.path.dirname(args.input),
    os.path.basename(args.input).replace("generations", "judged"),
)

with open(args.input) as f:
    rows = [json.loads(line) for line in f]

num_rollouts = len(rows[0]["responses"])
print(f"Total prompts: {len(rows)}, rollouts per prompt: {num_rollouts}", flush=True)

flat_pairs = []
for i, row in enumerate(rows):
    for j, resp in enumerate(row["responses"]):
        flat_pairs.append((i, j, row["adversarial"], resp))

total_pairs = len(flat_pairs)
print(f"Total sequences to classify: {total_pairs}", flush=True)

print(f"Loading {args.judge} with vLLM...", flush=True)
llm = LLM(model=args.judge, dtype="bfloat16", seed=args.seed,
          max_model_len=2560, enforce_eager=True)
tokenizer = llm.get_tokenizer()

sampling_params = SamplingParams(
    max_tokens=32,
    temperature=0,
)

print(f"Tokenizing {total_pairs} inputs...", flush=True)
all_token_ids = build_refusal_token_ids(tokenizer, flat_pairs)
prompts = [{"prompt_token_ids": ids} for ids in all_token_ids]

refusal_votes = {}
for chunk_start in range(0, total_pairs, args.chunk_size):
    chunk_end = min(chunk_start + args.chunk_size, total_pairs)
    outputs = llm.generate(prompts[chunk_start:chunk_end], sampling_params)
    chunk_pairs = flat_pairs[chunk_start:chunk_end]

    for idx, output in enumerate(outputs):
        row_idx, rollout_idx, _, _ = chunk_pairs[idx]
        result = output.outputs[0].text.strip()
        refusal_votes[(row_idx, rollout_idx)] = wildguard_refusal_parse(result)

    print(f"[refusal] Completed {chunk_end}/{total_pairs}", flush=True)

del llm, tokenizer, all_token_ids, prompts
torch.cuda.empty_cache()
gc.collect()

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as out_f:
    for i, row in enumerate(rows):
        rollout_judgments = []
        for j in range(len(row["responses"])):
            rollout_judgments.append({"refused": refusal_votes[(i, j)]})
        num_refused = sum(1 for rj in rollout_judgments if rj["refused"])
        out_row = {
            "adversarial": row["adversarial"],
            "responses": row["responses"],
            "rollout_judgments": rollout_judgments,
            "refusal_rate": num_refused / len(rollout_judgments),
            "compliance_rate": 1.0 - num_refused / len(rollout_judgments),
        }
        out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

# Summary
with open(OUTPUT_FILE) as f:
    final = [json.loads(line) for line in f]
n_has_compliant = sum(1 for r in final if r["compliance_rate"] > 0)
mean_refusal = sum(r["refusal_rate"] for r in final) / len(final) if final else 0.0

print(f"\n{'='*60}")
print(f"FINAL SUMMARY — adv_benign refusal judging")
print(f"{'='*60}")
print(f"Total prompts: {len(final)}")
print(f"Rollouts per prompt: {num_rollouts}")
print(f"Mean per-rollout refusal rate: {mean_refusal:.1%}")
print(f"Prompts with ≥1 non-refused rollout (eligible for training): {n_has_compliant}/{len(final)}")
print(f"\nSaved to {OUTPUT_FILE}")
