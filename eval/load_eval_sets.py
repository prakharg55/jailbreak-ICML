import argparse
import json
import os
from typing import Any, Dict, List

from datasets import load_dataset
import pandas as pd
from huggingface_hub import hf_hub_download

from dataset_configs import DATASET_CONFIGS, ensure_parent_dir


def save_jsonl(rows: List[Dict[str, Any]], path: str):
    ensure_parent_dir(path)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows to {path}")


def normalize_text(x):
    if x is None:
        return None
    if isinstance(x, list):
        return "\n".join(str(v) for v in x if v is not None).strip()
    return str(x).strip()


def load_wildjailbreak(output_path: str):
    eval_path = hf_hub_download(
        repo_id="allenai/wildjailbreak",
        filename="eval/eval.tsv",
        repo_type="dataset",
    )
    df = pd.read_csv(eval_path, sep="\t")

    rows = []
    for i, ex in df.iterrows():
        if ex.get("data_type") != "adversarial_harmful":
            continue

        prompt = normalize_text(ex.get("adversarial"))
        if not prompt:
            continue

        row_id = ex.get("id", f"wildjailbreak_eval_{i}")
        if pd.isna(row_id):
            row_id = f"wildjailbreak_eval_{i}"

        rows.append({
            "dataset": "wildjailbreak",
            "split": "eval",
            "id": row_id,
            "prompt": prompt,
            "prompt_label": "harmful",
            "metadata": {
                "data_type": ex.get("data_type"),
                "source_row": i,
            }
        })

    save_jsonl(rows, output_path)


def load_wildguardmix(output_path: str):
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    rows = []
    for i, ex in enumerate(ds):
        prompt = normalize_text(ex.get("prompt"))
        if not prompt:
            continue
        if ex.get("adversarial") is not True:
            continue
        if ex.get("prompt_harm_label") != "harmful":
            continue

        rows.append({
            "dataset": "wildguardmix",
            "split": "wildguardtest/test",
            "id": ex.get("id", f"wildguardmix_test_{i}"),
            "prompt": prompt,
            "prompt_label": "harmful",
            "metadata": {
                "adversarial": ex.get("adversarial"),
                "subcategory": ex.get("subcategory"),
                "prompt_harm_label": ex.get("prompt_harm_label"),
                "source_row": i,
            }
        })
    save_jsonl(rows, output_path)


def load_clearharm(output_path: str):
    candidate_configs = [None, "default", "pos", "rep40"]
    rows = []
    seen = set()

    for cfg in candidate_configs:
        try:
            if cfg is None:
                ds = load_dataset("AlignmentResearch/ClearHarm", split="train")
            else:
                ds = load_dataset("AlignmentResearch/ClearHarm", cfg, split="train")
        except Exception as e:
            print(f"Skipping ClearHarm config={cfg}: {e}")
            continue

        print(f"Loaded ClearHarm config={cfg} train split with {len(ds)} rows")

        for i, ex in enumerate(ds):
            prompt = None
            for key in ["prompt", "instruction", "instructions", "question", "input", "content", "text"]:
                if key in ex and ex[key]:
                    prompt = normalize_text(ex[key])
                    break

            if not prompt:
                continue

            label = None
            if "prompt_harm_label" in ex:
                label = ex["prompt_harm_label"]
            elif "clf_label" in ex:
                label = "harmful" if str(ex["clf_label"]).strip().lower() in {"1", "harmful"} else "unharmful"

            if label not in {None, "harmful"}:
                continue

            uniq = (cfg, prompt)
            if uniq in seen:
                continue
            seen.add(uniq)

            rows.append({
                "dataset": "clearharm",
                "split": "train",
                "id": ex.get("id", f"clearharm_{cfg}_train_{i}"),
                "prompt": prompt,
                "prompt_label": "harmful",
                "metadata": {
                    "config": cfg,
                    "source_row": i,
                }
            })

    if not rows:
        raise RuntimeError("No ClearHarm rows loaded from train split. Inspect one example locally to confirm field names.")

    save_jsonl(rows, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    output = args.output or cfg["output_file"]

    if args.dataset == "wildjailbreak":
        load_wildjailbreak(output)
    elif args.dataset == "wildguardmix":
        load_wildguardmix(output)
    elif args.dataset == "clearharm":
        load_clearharm(output)


if __name__ == "__main__":
    main()