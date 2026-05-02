# Code: Self-Mined Hardness for Safety Fine-Tuning

This is the anonymized code release that accompanies the workshop paper
*"Self-Mined Hardness for Safety Fine-Tuning"*. It implements the full
pipeline: (1) self-mining hard adversarial prompts from the target model's
own rollouts via a three-judge safety ensemble, (2) mining adversarially-framed
benign prompts via a refusal classifier, (3) building the five training
baselines (Hard, Random, Control, Hard-Mixed, Random-Mixed) under a
canonical prompt-to-target pairing, (4) LoRA fine-tuning, and (5) test-set
evaluation across three safety and three overrefusal benchmarks.

The paper reports results on `meta-llama/Meta-Llama-3-8B-Instruct` and
`meta-llama/Llama-3.2-3B-Instruct`. The code is model-agnostic — pick the
target via the `MODEL` variable in the `.sh` driver scripts.

## Layout

```
code/
├── requirements.txt           # pip freeze from the experiment environment
├── train/                     # numbered pipeline (1 → 7), run in order
│   ├── 1_load_data.{py,sh}        # download WildJailbreak partitions
│   ├── 2_analyze_data.py          # quick stats over the corpus
│   ├── 3_sample_data.py           # uniform sample of N_h adversarial-harmful prompts
│   ├── 4_generate_rollouts.{py,sh}    # K_h=64 rollouts per prompt at T=1
│   ├── 4b_generate_benign_rollouts.{py,sh}  # K_b=4 rollouts on adv_benign
│   ├── 5_judge_rollouts.{py,sh}       # 3-judge majority safety vote → harmful_rate
│   ├── 5b_judge_benign.{py,sh}        # WildGuard refusal classification
│   ├── 6_save_training_data.{py,sh}   # build all 5 baselines (hard/random/control + hard_mixed/random_mixed with 1:1 interleave)
│   └── 7_train.{py,sh}                # LoRA SFT per baseline, checkpointed every 50 prompts
└── eval/                      # test-set evaluation
    ├── data/                  # six held-out test sets (small JSONLs)
    ├── load_eval_sets.py      # rebuild data/*.jsonl from the public datasets
    ├── build_overrefusal_test_sets.py  # build the three overrefusal sets
    ├── dataset_configs.py     # Hugging Face dataset metadata
    ├── batch_generate.py      # generate for many (adapter × dataset) pairs in one go
    ├── batch_judge.py         # judge generations for a results dir
    ├── run_dataset_eval.py    # generate + judge + summarize for one dataset
    ├── summarize_eval.py      # ASR / refusal-rate summary JSON
    ├── run_test_eval.sh       # SLURM driver for full test-set evaluation
    ├── plot_paper.py          # per-model results table (.tex) + bar plots
    └── plot_summary.py        # 2x2 headline figure across both target models
```

## Setup

Python 3.10 with CUDA 12.x. Install dependencies:

```bash
pip install -r requirements.txt
```

Authenticate with Hugging Face for gated models/datasets
(`meta-llama/*`, `allenai/wildguard`, `allenai/wildjailbreak`,
`OpenSafetyLab/MD-Judge-v0.1`, `meta-llama/Llama-Guard-3-8B`):

```bash
huggingface-cli login
```

Set a checkpoint root (the `7_train.py` script writes adapter checkpoints
under `${CHECKPOINT_ROOT}/<MODEL_SHORT>/<DATA>/lora_r16/`):

```bash
export CHECKPOINT_ROOT=/path/to/your/scratch/checkpoints
```

The `.sh` files use SLURM directives (`#SBATCH`) — replace `YOUR_ACCOUNT`,
`YOUR_EMAIL`, and `YOUR_PARTITION` with your cluster's values, or run the
underlying `python3 ...` commands directly on a single node.

All scripts are written to be run from the project root (i.e., the parent
directory of `train/` and `eval/`).

## Pipeline

### 1. Self-mine hardness (one-time per target model)

The hard side (adversarial-harmful prompts) and the benign side
(adversarially-framed benign prompts) are independent and can run in parallel.

Hard side:

```bash
python3 train/1_load_data.py             # WildJailbreak partitions
python3 train/3_sample_data.py --n 5000  # N_h = 5,000 adv-harmful prompts
sbatch  train/4_generate_rollouts.sh     # K_h = 64 rollouts each, T=1
sbatch  train/5_judge_rollouts.sh        # 3-judge majority safety vote
```

Benign side:

```bash
sbatch train/4b_generate_benign_rollouts.sh   # K_b = 4 rollouts on adv_benign
sbatch train/5b_judge_benign.sh               # WildGuard refusal classifier
```

### 2. Build the five training datasets

```bash
sbatch train/6_save_training_data.sh
```

This writes `train_hard_*.jsonl`, `train_random_*.jsonl`,
`train_control_*.jsonl`, `train_hard_mixed_*.jsonl`,
`train_random_mixed_*.jsonl` into `train/results/`. All five files share
the same canonical (prompt → safe-response) pairing for the eligible pool;
mixed files interleave the eligible pool 1:1 with the eligible adv-benign
set in strict alternating order.

### 3. Train all five LoRA adapters

```bash
for B in hard random control hard_mixed random_mixed; do
  sbatch --job-name=train_${B} --export=ALL,BASELINE=${B} train/7_train.sh
done
```

Hyperparameters are in `train/7_train.py` and `train/7_train.sh`
(LoRA r=16, α=32, dropout=0.05, lr=1e-4, effective batch 10, 1 epoch,
cosine LR with 0.03 warmup, AdamW, bf16, seed=55, checkpoint every 5
optimizer steps = every 50 prompts).

### 4. Build the overrefusal test sets and evaluate

The WildJailbreak adv-benign and ClearHarm test sets are direct copies of
the public eval splits. The two WildGuardMix test sets are constructed
from the WildGuardMix `test` split; rebuild them once with:

```bash
python3 eval/build_overrefusal_test_sets.py
```

(Or just use the JSONL files already in `eval/data/`.)

To generate and judge at the 50%-of-pool checkpoints reported in the paper:

```bash
sbatch eval/run_test_eval.sh
```

By default this targets `Meta-Llama-3-8B-Instruct` and the checkpoints
`step 120` (pure regime, 50% of 8B's 2,388-prompt eligible pool, rounded
up to the nearest 50-prompt save point) and `step 240` (mixed regime,
2,400 = 2 × 1,200 prompts). The control comparison for the mixed regime
is taken at `step 235` (the closest matched-compute control checkpoint to
2,400). Edit `MODEL`, `STEP_PURE`, `STEP_MIXED` at the top of
`run_test_eval.sh` to reproduce on `Llama-3.2-3B-Instruct` (50% of 2,488
prompts → step 125 / step 250).

### 5. Plots and table

```bash
python3 eval/plot_paper.py     # per-model bar plots + results_table.tex
python3 eval/plot_summary.py   # 2x2 headline figure across both models
```

Outputs land under `eval/plots_paper/<MODEL_SHORT>/`.

## Reproducibility

- Seed 55 is used throughout (rollout sampling, training, eval generation).
- The `T=1` sampling in steps 4 / 4b is the only non-deterministic step
  in the pipeline; the canonical (prompt → safe target) mapping fixes the
  result of any random tie-breaking once `5_judge_rollouts.py` and
  `5b_judge_benign.py` have run.
- All judges (WildGuard, MD-Judge v0.1, Llama-Guard-3-8B for safety;
  WildGuard alone for refusal classification) run with greedy decoding.
- Test-set eval generations use greedy decoding (one rollout per prompt).

## What this release does and does not include

Includes: the full self-mining → training → evaluation pipeline, the six
held-out test set JSONLs, plotting scripts, and the full `pip freeze`.

Does not include: training data (regenerate via the pipeline; they are
large), checkpoint weights, intermediate generation/judging logs, or
results files (regenerate by running the pipeline).

The release was anonymized prior to submission — author-identifying SLURM
account/email fields have been replaced with `YOUR_ACCOUNT` / `YOUR_EMAIL`
placeholders.
