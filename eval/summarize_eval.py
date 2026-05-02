import argparse
import json
import os
from statistics import mean


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize_file(path):
    rows = load_jsonl(path)
    if not rows:
        return None

    first = rows[0]
    metric_type = first.get("metric_type")

    if metric_type not in {"jailbreak_rate", "overrefusal_rate"}:
        return None

    summary = {
        "file": path,
        "dataset": first.get("dataset"),
        "metric_type": metric_type,
        "num_prompts": len(rows),
        "base_model": first.get("generation_config", {}).get("base_model"),
        "adapter": first.get("generation_config", {}).get("adapter"),
    }

    if metric_type == "jailbreak_rate":
        summary["mean_harmful_rate"] = mean(r["harmful_rate"] for r in rows)
        summary["mean_unharmful_rate"] = mean(r["unharmful_rate"] for r in rows)
        summary["attack_success_rate"] = (
            sum(r["harmful_rate"] > 0 for r in rows) / len(rows)
        )
    else:
        summary["mean_refusal_rate"] = mean(r["refusal_rate"] for r in rows)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        paths = [
            os.path.join(args.input, name)
            for name in sorted(os.listdir(args.input))
            if name.endswith("_judged.jsonl")
        ]
    else:
        paths = [args.input]

    summaries = []
    for path in paths:
        try:
            s = summarize_file(path)
            if s is not None:
                summaries.append(s)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Saved summary to {args.output}")

    for s in summaries:
        print("=" * 80)
        print(s["file"])
        print(f"dataset={s['dataset']} prompts={s['num_prompts']} metric={s['metric_type']}")
        print(f"base_model={s['base_model']}")
        print(f"adapter={s['adapter']}")
        if s["metric_type"] == "jailbreak_rate":
            print(f"attack_success_rate={s['attack_success_rate']:.4f}")
            print(f"mean_harmful_rate={s['mean_harmful_rate']:.4f}")
            print(f"mean_unharmful_rate={s['mean_unharmful_rate']:.4f}")
        else:
            print(f"mean_refusal_rate={s['mean_refusal_rate']:.4f}")


if __name__ == "__main__":
    main()