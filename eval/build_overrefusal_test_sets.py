"""Build the three additional overrefusal test sets from HuggingFace datasets.

Outputs under eval/data/:
  wildjailbreak_adv_benign_eval.jsonl  — 210 adversarial_benign prompts (WildJailbreak eval split)
  wildguardmix_adv_benign_eval.jsonl   — 455 adversarial_benign prompts (WildGuardMix test split)
  wildguardmix_benign_eval.jsonl       — 490 vanilla_benign prompts (WildGuardMix test split)

These complement the existing safety test sets (wildjailbreak_eval.jsonl = 2000 adv_harmful,
wildguardmix_eval.jsonl = 341 adv_harmful) by adding the matched overrefusal distributions
from the same source datasets.

Schema matches the existing *_eval.jsonl files:
  {dataset, split, id, prompt, prompt_label, metadata}

Run from project root: python3 eval/build_overrefusal_test_sets.py
"""

import json
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows to {path}")


def build_wildjailbreak_adv_benign():
    """WildJailbreak eval split has 210 adversarial_benign prompts."""
    print("Loading WildJailbreak eval split...", flush=True)
    path = hf_hub_download(repo_id="allenai/wildjailbreak", filename="eval/eval.tsv", repo_type="dataset")
    df = pd.read_csv(path, sep="\t")
    adv_ben = df[df["data_type"] == "adversarial_benign"]
    print(f"  {len(adv_ben)} adversarial_benign rows")

    rows = []
    for i, row in enumerate(adv_ben.itertuples()):
        rows.append({
            "dataset": "wildjailbreak_adv_benign",
            "split": "eval",
            "id": f"wildjailbreak_adv_benign_eval_{i}",
            "prompt": row.adversarial.strip(),
            "prompt_label": "benign",
            "metadata": {"data_type": "adversarial_benign", "source": "wildjailbreak_eval_split"},
        })
    save_jsonl(rows, "eval/data/wildjailbreak_adv_benign_eval.jsonl")


def build_wildguardmix_overrefusal():
    """WildGuardMix test split has both adversarial_benign and vanilla_benign.

    Partition by (adversarial, prompt_harm_label):
      - (True, unharmful)  → adversarial_benign  (455)
      - (False, unharmful) → vanilla_benign      (490)
    """
    print("Loading WildGuardMix test split...", flush=True)
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")

    adv_benign_rows = []
    van_benign_rows = []
    for i, r in enumerate(ds):
        if r["prompt_harm_label"] != "unharmful":
            continue
        if r["adversarial"] is True:
            adv_benign_rows.append((i, r))
        elif r["adversarial"] is False:
            van_benign_rows.append((i, r))

    print(f"  {len(adv_benign_rows)} adversarial_benign + {len(van_benign_rows)} vanilla_benign")

    # adversarial_benign
    rows = []
    for local_i, (orig_i, r) in enumerate(adv_benign_rows):
        rows.append({
            "dataset": "wildguardmix_adv_benign",
            "split": "eval",
            "id": f"wildguardmix_adv_benign_eval_{local_i}",
            "prompt": r["prompt"].strip(),
            "prompt_label": "benign",
            "metadata": {
                "orig_idx": orig_i,
                "adversarial": True,
                "prompt_harm_label": "unharmful",
                "source": "wildguardmix_test_split",
            },
        })
    save_jsonl(rows, "eval/data/wildguardmix_adv_benign_eval.jsonl")

    # vanilla_benign (plain benign)
    rows = []
    for local_i, (orig_i, r) in enumerate(van_benign_rows):
        rows.append({
            "dataset": "wildguardmix_benign",
            "split": "eval",
            "id": f"wildguardmix_benign_eval_{local_i}",
            "prompt": r["prompt"].strip(),
            "prompt_label": "benign",
            "metadata": {
                "orig_idx": orig_i,
                "adversarial": False,
                "prompt_harm_label": "unharmful",
                "source": "wildguardmix_test_split",
            },
        })
    save_jsonl(rows, "eval/data/wildguardmix_benign_eval.jsonl")


def main():
    build_wildjailbreak_adv_benign()
    build_wildguardmix_overrefusal()


if __name__ == "__main__":
    main()
