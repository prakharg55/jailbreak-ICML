import os

DATASET_CONFIGS = {
    "wildjailbreak": {
        "hf_name": "allenai/wildjailbreak",
        "mode": "safety",
        "default_split": "eval",
        "output_file": "data/wildjailbreak_eval.jsonl",
    },
    "wildguardmix": {
        "hf_name": "allenai/wildguardmix",
        "hf_config": "wildguardtest",
        "mode": "safety",
        "default_split": "test",
        "output_file": "data/wildguardmix_eval.jsonl",
    },
    "clearharm": {
        "hf_name": "AlignmentResearch/ClearHarm",
        "mode": "safety",
        "output_file": "data/clearharm_eval.jsonl",
    },
}

BASELINE_CONFIGS = {
    "hard": {},
    "random": {},
    "control": {},
}


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)