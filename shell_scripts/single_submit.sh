#!/bin/bash

#SBATCH --job-name=train_npi
#SBATCH --partition=h200-batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=7-00:00:00
#SBATCH --output=train_npi-%j.out

cd ..
uv run train_single_subject.py configs/single_subject_config.yaml
# uv run torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc-per-node 2 \
#     train_single_subject.py configs/single_subject_config.yaml
