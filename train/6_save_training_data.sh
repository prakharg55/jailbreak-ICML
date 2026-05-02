#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=save_training_data
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=YOUR_PARTITION
#SBATCH --mem=64G
#SBATCH --output=train/results/save_training_data_llama8b.log

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="meta-llama/Llama-3.2-3B-Instruct"

MODEL_SHORT="${MODEL##*/}"

python3 train/6_save_training_data.py \
    --model "$MODEL" \
    --input "train/results/judged_${MODEL_SHORT}_n5000_r64.jsonl" \
    --adv-benign-input "train/results/judged_benign_${MODEL_SHORT}_n3000_r4.jsonl"
    # --baselines hard random
    # --baselines control --n 5000

