import argparse
import os
import subprocess


def run(cmd: str):
    print(f"\n[RUN] {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--gen-chunk-size", type=int, default=256)
    parser.add_argument("--judge-chunk-size", type=int, default=32000)
    args = parser.parse_args()

    model_name = os.path.basename(args.base_model.rstrip("/"))
    if args.results_dir is None:
        args.results_dir = os.path.join("eval", "results", model_name)

    os.makedirs(args.results_dir, exist_ok=True)

    if args.dataset == "wildjailbreak":
        eval_input = "eval/data/wildjailbreak_eval.jsonl"
        mode = "safety"
    elif args.dataset == "wildguardmix":
        eval_input = "eval/data/wildguardmix_eval.jsonl"
        mode = "safety"
    elif args.dataset == "clearharm":
        eval_input = "eval/data/clearharm_eval.jsonl"
        mode = "safety"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    load_cmd = f"python3 eval/load_eval_sets.py --dataset '{args.dataset}'"
    run(load_cmd)

    gen_out = os.path.join(dataset_dir, f"{args.baseline_name}_generations.jsonl")
    judge_out = os.path.join(dataset_dir, f"{args.baseline_name}_judged.jsonl")

    gen_cmd = (
        f"python3 eval/generate_rollouts.py "
        f"--input '{eval_input}' "
        f"--base-model '{args.base_model}' "
        f"--output '{gen_out}' "
        f"--num-rollouts {args.num_rollouts} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--temperature {args.temperature} "
        f"--top-p {args.top_p} "
        f"--seed {args.seed} "
        f"--chunk-size {args.gen_chunk_size}"
    )

    if args.adapter:
        gen_cmd += f" --adapter '{args.adapter}'"

    if args.max_examples is not None:
        gen_cmd += f" --max-examples {args.max_examples}"

    run(gen_cmd)

    if mode == "safety":
        run(
            f"python3 eval/judge_rollouts.py "
            f"--input '{gen_out}' "
            f"--output '{judge_out}' "
            f"--mode safety "
            f"--seed {args.seed} "
            f"--chunk-size {args.judge_chunk_size}"
        )
    else:
        run(
            f"python3 eval/judge_rollouts.py "
            f"--input '{gen_out}' "
            f"--output '{judge_out}' "
            f"--mode overrefusal "
            f"--seed {args.seed} "
            f"--chunk-size {args.judge_chunk_size}"
        )

    summary_out = os.path.join(dataset_dir, f"{args.baseline_name}_summary.json")
    run(
        f"python3 eval/summarize_eval.py "
        f"--input '{judge_out}' "
        f"--output '{summary_out}'"
    )


if __name__ == "__main__":
    main()