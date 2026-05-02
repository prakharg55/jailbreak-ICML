import json
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="train/data/wildjailbreak/train.jsonl")
parser.add_argument("--n", type=int, default=None, help="Number of samples (default: use all)")
parser.add_argument("--seed", type=int, default=55)
args = parser.parse_args()

path = args.input
n = args.n
seed = args.seed

with open(path) as f:
    rows = [json.loads(line) for line in f]

# Filter to adversarial_harmful only, with non-empty prompt
rows = [r for r in rows if r.get("data_type") == "adversarial_harmful" and r.get("adversarial")]
print(f"Total adversarial_harmful rows: {len(rows)}")

random.seed(seed)
if n is None:
    sampled = rows
    n = len(rows)
else:
    sampled = random.sample(rows, n)

out_path = path.replace(".jsonl", f"_{n}.jsonl")
with open(out_path, "w") as f:
    for row in sampled:
        f.write(json.dumps(row) + "\n")

print(f"Saved {n} samples to {out_path}")
