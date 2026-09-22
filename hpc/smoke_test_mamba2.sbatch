#!/bin/bash
#SBATCH --job-name=mamba2-smoke
#SBATCH --account=def-brodbeck
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/zhanz720/library_speech/log/%x-%j.out
#SBATCH --error=/home/zhanz720/library_speech/log/%x-%j.err

set -euo pipefail

module load python/3.11
source /home/zhanz720/envs/librispeech/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=7

cd /home/zhanz720/library_speech
srun python -u scripts/smoke_test_mamba2.py
