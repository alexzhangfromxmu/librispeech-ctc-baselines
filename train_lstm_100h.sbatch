#!/bin/bash
#SBATCH --job-name=lstm-ctc-100h
#SBATCH --account=def-brodbeck
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhanz720/library_speech/logs/%x-%j.out
#SBATCH --error=/home/zhanz720/library_speech/logs/%x-%j.err

set -euo pipefail

module load python/3.11
source /home/zhanz720/envs/librispeech/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

CODE_DIR=/home/zhanz720/library_speech
DATASET_DIR=/scratch/zhanz720/library_speech/dataset
OUTPUT_DIR=/home/zhanz720/library_speech/stage2_outputs_100h_h100

cd "${CODE_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Python: $(command -v python)"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u stage2_lstm_ctc.py \
  --dataset-root "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --full-train \
  --full-dev \
  --epochs 30 \
  --batch-size 8 \
  --hidden-size 256 \
  --lstm-layers 3 \
  --dropout 0.2 \
  --learning-rate 0.001 \
  --early-stopping-patience 7 \
  --lr-patience 2 \
  --num-workers 0 \
  --seed 7

echo "Finished: $(date --iso-8601=seconds)"
