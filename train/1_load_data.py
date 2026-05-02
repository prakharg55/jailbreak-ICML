import os
import pandas as pd
from huggingface_hub import hf_hub_download

os.makedirs("train/data/wildjailbreak", exist_ok=True)

train_path = hf_hub_download(repo_id="allenai/wildjailbreak", filename="train/train.tsv", repo_type="dataset")
train_df = pd.read_csv(train_path, sep="\t")
train_df.to_json("train/data/wildjailbreak/train.jsonl", orient="records", lines=True)
print(f"Train: {len(train_df)} rows, columns: {train_df.columns.tolist()}")

# eval_path = hf_hub_download(repo_id="allenai/wildjailbreak", filename="eval/eval.tsv", repo_type="dataset")
# eval_df = pd.read_csv(eval_path, sep="\t")
# eval_df.to_json("data/wildjailbreak/eval.jsonl", orient="records", lines=True)
# print(f"Eval: {len(eval_df)} rows, columns: {eval_df.columns.tolist()}")
