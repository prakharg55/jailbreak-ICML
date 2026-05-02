#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=test_eval
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=YOUR_PARTITION
#SBATCH --mem=96G
#SBATCH --output=eval/results/test_eval_llama8b.log

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="meta-llama/Llama-3.2-3B-Instruct"  # uncomment for the 3B run

# Paper uses the "50% of hardness-ranked pool" checkpoint for each regime:
#   Pure  (N=1200 prompts = 2388/2): step 120  → τ ≈ 0.094
#   Mixed (N=2400 total = 2*1200):   step 240  → same 1200 hard prompts + 1200 adv_benign
# Control baseline trained for a full 239 steps (no mixed variant), so its closest
# matched-compute checkpoint for the mixed plot is step 235 (trained on 2350 benign).
# We generate and judge at both step 120 and step 235 for control so each plot
# has a matched reference line.

STEP_PURE="120"
STEP_MIXED="240"
STEP_CONTROL_MIXED="235"  # closest to STEP_MIXED among saved control ckpts

MODEL_SHORT="${MODEL##*/}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT:-./checkpoints}/checkpoints/${MODEL_SHORT}"
RESULTS_DIR="eval/results/${MODEL_SHORT}"

# Phase 1: Generate for base + selected checkpoints × all 6 test datasets
# Safety:
#   wildjailbreak (2000)   — adversarial-harmful
#   wildguardmix (341)     — adversarial-harmful
#   clearharm (716)        — clearly-harmful
# Overrefusal:
#   wildjailbreak_adv_benign (210)  — adversarially-framed benign
#   wildguardmix_adv_benign (455)   — adversarial-yet-unharmful
#   wildguardmix_benign (490)       — plain benign
echo "===== PHASE 1: GENERATION ====="
python3 eval/batch_generate.py \
    --base-model "$MODEL" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --adapters \
        "hard:checkpoint-${STEP_PURE}" \
        "random:checkpoint-${STEP_PURE}" \
        "control:checkpoint-${STEP_PURE}" \
        "control:checkpoint-${STEP_CONTROL_MIXED}" \
        "hard_mixed:checkpoint-${STEP_MIXED}" \
        "random_mixed:checkpoint-${STEP_MIXED}"

# Phase 2: Judge all generated files
# batch_judge routes safety datasets to the 3-judge ensemble and overrefusal
# datasets to the WildGuard refusal classifier, with per-judge
# "load once, classify all" batching to amortize judge load time.
echo "===== PHASE 2: JUDGING ====="
python3 eval/batch_judge.py \
    --results-dir "$RESULTS_DIR"

echo "===== TEST EVAL COMPLETE ====="

