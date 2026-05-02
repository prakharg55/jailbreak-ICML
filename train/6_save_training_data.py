import argparse
import json
import random
import gc
import os
import torch

# Patch transformers 5.3+ compatibility with vLLM 0.11
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="train/results/judged_Meta-Llama-3-8B-Instruct_n5000_r64.jsonl")
parser.add_argument(
    "--vanilla-data",
    default="train/data/wildjailbreak/train.jsonl",
    help="Raw dataset with vanilla prompts for control baseline",
)
parser.add_argument(
    "--model",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Model for generating control rollouts",
)
parser.add_argument(
    "--n",
    type=int,
    default=None,
    help="Number of adversarial prompts to select (default: all eligible)",
)
parser.add_argument(
    "--baselines",
    nargs="+",
    default=["hard", "random", "control"],
    choices=["hard", "random", "control"],
    help="Which pure baselines to generate (default: all three)",
)
parser.add_argument(
    "--adv-benign-input",
    default=None,
    help="judged_benign_*.jsonl from 5b_judge_benign.py. If provided, hard_mixed and "
         "random_mixed interleave with adv_benign (non-refused rollouts) instead of control. "
         "Pure `control` baseline is unaffected.",
)
parser.add_argument("--seed", type=int, default=55)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

needs_judged_input = "hard" in args.baselines or "random" in args.baselines


def save_jsonl(data, path):
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def make_output_path(suffix):
    return os.path.join(
        os.path.dirname(args.input),
        os.path.basename(args.input).replace("judged", suffix),
    )


def pick_safe_response(row):
    """Random non-harmful rollout response for one row. Called once per unique prompt
    when building the canonical mapping below — never called per-baseline — so the
    same prompt always yields the same safe_response across hard, random, hard_mixed,
    random_mixed."""
    safe_indices = [
        j for j, rj in enumerate(row["rollout_judgments"])
        if rj["majority_unharmful"]
    ]
    if not safe_indices:
        raise ValueError("Row has no majority_unharmful rollout.")
    return row["responses"][random.choice(safe_indices)]


def pick_compliant_response(row):
    """Random non-refused rollout response for one adv_benign row. Called once per
    unique prompt when building the canonical mapping — mirrors pick_safe_response."""
    compliant_indices = [
        j for j, rj in enumerate(row["rollout_judgments"])
        if not rj["refused"]
    ]
    if not compliant_indices:
        raise ValueError("Row has no non-refused rollout.")
    return row["responses"][random.choice(compliant_indices)]


# Canonical mappings populated once below, before any baseline is built.
hard_prompt_to_safe = {}
benign_prompt_to_compliant = {}


def make_adv_example(row):
    return {
        "adversarial": row["adversarial"],
        "safe_response": hard_prompt_to_safe[row["adversarial"]],
        "harmful_rate": row["harmful_rate"],
        "unharmful_rate": row["unharmful_rate"],
        "source": "adversarial",
    }


def make_adv_benign_example(row):
    return {
        "adversarial": row["adversarial"],
        "safe_response": benign_prompt_to_compliant[row["adversarial"]],
        "source": "adv_benign",
    }


def interleave_equal(a, b):
    if len(a) != len(b):
        raise ValueError(f"Cannot interleave unequal lengths: {len(a)} vs {len(b)}")
    out = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    return out


eligible = []
hard_training_data = []
random_training_data = []
control_training_data = []
n_selected = args.n

# ---- Load and process judged rollouts (only needed for hard/random) ----
if needs_judged_input:
    with open(args.input) as f:
        rows = [json.loads(line) for line in f]

    print(f"Total prompts: {len(rows)}")
    print()

    rows.sort(key=lambda r: r["harmful_rate"], reverse=True)

    no_harmful = []
    no_safe = []

    for row in rows:
        harmful_rate = row["harmful_rate"]
        safe_rate = row["unharmful_rate"]

        if harmful_rate == 0:
            no_harmful.append(row)
        elif safe_rate == 0:
            no_safe.append(row)
        else:
            eligible.append(row)

    print(f"Prompts with no harmful rollouts (model never jailbroken):  {len(no_harmful)}")
    print(f"Prompts with no safe rollouts (can't pair with clean response): {len(no_safe)}")
    print(f"Eligible prompts (have both harmful and safe rollouts):        {len(eligible)}")
    print()

    if no_safe:
        print("--- DROPPED (no safe rollout to use as training target) ---")
        for i, row in enumerate(no_safe):
            print(
                f"  {i+1}. harmful={row['harmful_rate']:.0%} safe={row['unharmful_rate']:.0%} | "
                f"{row['adversarial'][:100]}"
            )
        print()

    if n_selected is not None:
        selected_hard_rows = eligible[:n_selected]
    else:
        selected_hard_rows = eligible

    print(f"Selected {len(selected_hard_rows)} hard prompts for training")
    print()

    # Canonical prompt→safe_response mapping. Built once from `eligible` so that
    # any downstream baseline (hard, random, hard_mixed, random_mixed) pairing the
    # same adversarial prompt gets the *same* safe_response — no independent re-rolls.
    for row in eligible:
        hard_prompt_to_safe[row["adversarial"]] = pick_safe_response(row)

    hard_training_data = [make_adv_example(row) for row in selected_hard_rows]
    n_selected = len(hard_training_data)

    print("--- SELECTED HARD PROMPTS (sorted by harmful rate) ---")
    for i, row in enumerate(hard_training_data):
        print(f"{i+1}. harmful={row['harmful_rate']:.0%} safe={row['unharmful_rate']:.0%}")
        print(f"   PROMPT: {row['adversarial'][:150]}")
        print(f"   SAFE RESPONSE: {row['safe_response'][:150]}")
        print()

if n_selected is None:
    print("Error: --n is required when running only --baselines control")
    raise SystemExit(1)

summary = []

# ---- 1) Hard baseline: top N by harmful rate, hardest first ----
if "hard" in args.baselines:
    hard_training_data.sort(key=lambda r: r["harmful_rate"], reverse=True)
    hard_file = make_output_path("train_hard")
    save_jsonl(hard_training_data, hard_file)
    print(f"Saved {len(hard_training_data)} hard-mined training examples to {hard_file}")
    summary.append(f"  Hard-mined (top {n_selected} by harmful rate): {hard_file}")

# ---- 2) Random baseline: N random eligible prompts, random order ----
if "random" in args.baselines:
    if n_selected <= len(eligible):
        random_selected_rows = random.sample(eligible, n_selected)
    else:
        random_selected_rows = list(eligible)

    random.shuffle(random_selected_rows)
    random_training_data = [make_adv_example(row) for row in random_selected_rows]

    random_file = make_output_path("train_random")
    save_jsonl(random_training_data, random_file)
    print(f"Saved {len(random_training_data)} random baseline training examples to {random_file}")
    summary.append(f"  Random baseline ({n_selected} random eligible): {random_file}")

# ---- 3) Control baseline: vanilla prompts + one model generation each ----
if "control" in args.baselines:
    print("Loading vanilla prompts for control baseline...", flush=True)
    with open(args.vanilla_data) as f:
        vanilla_rows = [json.loads(line) for line in f]

    vanilla_rows = [
        r for r in vanilla_rows
        if r.get("data_type") == "vanilla_benign" and r.get("vanilla")
    ]
    print(f"Total vanilla prompts: {len(vanilla_rows)}")

    if n_selected <= len(vanilla_rows):
        control_selected = random.sample(vanilla_rows, n_selected)
    else:
        control_selected = vanilla_rows

    print(f"Loading {args.model} with vLLM to generate control rollouts...", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=2560,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=1.0,
        n=1,
        seed=args.seed,
    )

    prompts = []
    for row in control_selected:
        messages = [{"role": "user", "content": row["vanilla"]}]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts.append({"prompt_token_ids": ids})

    print(f"Generating control rollouts: {len(control_selected)} prompts", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        control_training_data.append({
            "prompt": control_selected[i]["vanilla"],
            "response": response,
            "source": "control",
        })

    print(f"  Control {len(control_training_data)}/{len(control_selected)} generated", flush=True)

    del llm, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    control_file = make_output_path("train_control")
    save_jsonl(control_training_data, control_file)
    print(
        f"Saved {len(control_training_data)} control baseline "
        f"(vanilla prompts + model generations) to {control_file}"
    )
    summary.append(f"  Control baseline ({len(control_training_data)} vanilla prompts): {control_file}")

# ---- 4) Load adv_benign data (used by mixed baselines instead of control) ----
# Pure `control` baseline (vanilla_benign + one target-model rollout) is left
# untouched above. The benign *half* of hard_mixed / random_mixed switches to
# adversarial_benign prompts + a non-refused target-model rollout per prompt.
adv_benign_training_data = []
if args.adv_benign_input:
    print(f"\nLoading adv_benign judged file: {args.adv_benign_input}", flush=True)
    with open(args.adv_benign_input) as f:
        benign_rows = [json.loads(line) for line in f]

    eligible_benign = [
        r for r in benign_rows
        if any(not rj["refused"] for rj in r["rollout_judgments"])
    ]
    mean_refusal = (
        sum(r["refusal_rate"] for r in benign_rows) / len(benign_rows)
        if benign_rows else 0.0
    )
    print(f"  adv_benign total: {len(benign_rows)}")
    print(f"  mean per-rollout refusal rate: {mean_refusal:.1%}")
    print(f"  eligible (≥1 non-refused rollout): {len(eligible_benign)}")

    if len(eligible_benign) < n_selected:
        raise SystemExit(
            f"Need {n_selected} adv_benign survivors for 1:1 mix with hard, "
            f"but only {len(eligible_benign)} have a non-refused rollout. "
            f"Re-run train/4b_generate_benign_rollouts.py with a larger --n or --num_rollouts."
        )

    # Canonical benign prompt → compliant_response mapping. Built once so both
    # hard_mixed and random_mixed see the same (adv_benign prompt, response) pairings.
    for row in eligible_benign:
        benign_prompt_to_compliant[row["adversarial"]] = pick_compliant_response(row)

    # Sample exactly n_selected survivors (random subset if oversampled) so the
    # benign half of the mixed files is exactly matched in count to the hard half.
    selected_benign_rows = random.sample(eligible_benign, n_selected)
    adv_benign_training_data = [make_adv_benign_example(row) for row in selected_benign_rows]
    print(f"  using {len(adv_benign_training_data)} adv_benign examples for mixed baselines")

# ---- 5) Mixed baselines: prefix-balanced interleaving with adv_benign ----
# hard_mixed prefix property:
#   first 2k examples = first k hard examples from hard baseline (in desc harmful_rate order)
#                       + first k adv_benign examples
if "hard" in args.baselines and adv_benign_training_data:
    hard_training_data.sort(key=lambda r: r["harmful_rate"], reverse=True)

    if len(hard_training_data) != len(adv_benign_training_data):
        raise ValueError(
            f"hard / adv_benign length mismatch for hard_mixed: "
            f"{len(hard_training_data)} vs {len(adv_benign_training_data)}"
        )

    mixed_hard = interleave_equal(hard_training_data, adv_benign_training_data)

    mixed_hard_file = make_output_path("train_hard_mixed")
    save_jsonl(mixed_hard, mixed_hard_file)
    print(
        f"Saved {len(mixed_hard)} hard+adv_benign mixed training examples "
        f"(strict alternating order) to {mixed_hard_file}"
    )
    summary.append(
        f"  Hard+AdvBenign mixed ({len(mixed_hard)} = "
        f"{len(hard_training_data)} hard + {len(adv_benign_training_data)} adv_benign): "
        f"{mixed_hard_file}"
    )

# random_mixed prefix property:
#   first 2k examples = first k random examples + first k adv_benign examples
if "random" in args.baselines and adv_benign_training_data:
    if len(random_training_data) != len(adv_benign_training_data):
        raise ValueError(
            f"random / adv_benign length mismatch for random_mixed: "
            f"{len(random_training_data)} vs {len(adv_benign_training_data)}"
        )

    mixed_random = interleave_equal(random_training_data, adv_benign_training_data)

    mixed_random_file = make_output_path("train_random_mixed")
    save_jsonl(mixed_random, mixed_random_file)
    print(
        f"Saved {len(mixed_random)} random+adv_benign mixed training examples "
        f"(strict alternating order) to {mixed_random_file}"
    )
    summary.append(
        f"  Random+AdvBenign mixed ({len(mixed_random)} = "
        f"{len(random_training_data)} random + {len(adv_benign_training_data)} adv_benign): "
        f"{mixed_random_file}"
    )

print()
print("=== SUMMARY ===")
for line in summary:
    print(line)