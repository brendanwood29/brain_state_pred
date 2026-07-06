#!/bin/bash

#SBATCH --job-name=train_npi
#SBATCH --partition=h200-batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=7-00:00:00
#SBATCH --output=train_npi-%j.out

cd ..
uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node 2 \
    train.py configs/config.yaml
