#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=load_data
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=train/results/load_data.log

python3 train/1_load_data.py

