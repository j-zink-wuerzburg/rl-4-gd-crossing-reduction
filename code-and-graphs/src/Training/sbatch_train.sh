#!/bin/bash
#SBATCH -J ppo_lcr
#SBATCH --nodelist=gpu-amd
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out


set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

source /storage/home/brand/RLGD/SmartGD/venv2/bin/activate

date
echo "== SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "== Hostname: $(hostname)"
echo "== Python: $(which python)  ($(python -V))"

srun --ntasks=1 --cpu-bind=cores python -u train_Agent_new.py


echo "== Done"
date