import argparse
import json
import os
import random

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
    print(f"Saved {len(rows)} rows to {path}")


def build_chat_text(tokenizer, prompt, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if system_prompt:
        return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(args.input)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]

    print(f"Loading {args.base_model} with vLLM...", flush=True)
    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=2560,
        enforce_eager=True,
    )
    if args.adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.num_rollouts,
        seed=args.seed,
    )

    prompts = []
    for row in rows:
        text = build_chat_text(tokenizer, row["prompt"], args.system_prompt)
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts.append({"prompt_token_ids": ids})

    lora_request = None
    if args.adapter:
        lora_request = LoRARequest("eval_adapter", 1, args.adapter)

    output_rows = []
    total = len(rows)

    for chunk_start in range(0, total, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, total)
        chunk_prompts = prompts[chunk_start:chunk_end]
        chunk_rows = rows[chunk_start:chunk_end]

        outputs = llm.generate(
            chunk_prompts,
            sampling_params,
            lora_request=lora_request,
        )

        for row, output in zip(chunk_rows, outputs):
            responses = [o.text.strip() for o in output.outputs]
            out_row = dict(row)
            out_row["responses"] = responses
            out_row["generation_config"] = {
                "base_model": args.base_model,
                "adapter": args.adapter,
                "num_rollouts": args.num_rollouts,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            }
            output_rows.append(out_row)

        print(f"Completed {chunk_end}/{total} prompts", flush=True)

    save_jsonl(output_rows, args.output)


if __name__ == "__main__":
    main()