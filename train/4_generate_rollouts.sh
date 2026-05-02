#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=generate_rollouts
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=YOUR_PARTITION
#SBATCH --mem=64G
#SBATCH --output=train/results/generate_rollouts_llama8b.log

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="meta-llama/Llama-3.2-3B-Instruct"

python3 train/4_generate_rollouts.py --model "$MODEL"

