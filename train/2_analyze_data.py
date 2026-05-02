import argparse
import json
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="train/data/wildjailbreak/train.jsonl")
args = parser.parse_args()

path = args.input

records = [json.loads(line) for line in open(path)]
print(f"Total examples: {len(records)}\n")

for col in ["data_type"]:
    counts = Counter(r[col] for r in records)
    print(f"--- {col} ({len(counts)} unique values) ---")
    for val, count in counts.most_common():
        print(f"  {val}: {count}")
    print()
