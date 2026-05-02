#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=train_llama8b
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=train/results/%x.log

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="meta-llama/Llama-3.2-3B-Instruct"

# BASELINE is passed in via sbatch --export, e.g.:
#   sbatch --job-name=train_llama8b_hard --export=ALL,BASELINE=hard train/7_train.sh
# Valid: hard | random | control | hard_mixed | random_mixed
: "${BASELINE:?Set BASELINE env var (hard|random|control|hard_mixed|random_mixed)}"

MODEL_SHORT="${MODEL##*/}"

python3 train/7_train.py \
    --model "$MODEL" \
    --input "train/results/train_${BASELINE}_${MODEL_SHORT}_n5000_r64.jsonl"

