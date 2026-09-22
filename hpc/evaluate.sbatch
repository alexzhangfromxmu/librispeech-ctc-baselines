#!/bin/bash
#SBATCH --job-name=ctc-evaluate
#SBATCH --account=def-brodbeck
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhanz720/library_speech/log/%x-%j.out
#SBATCH --error=/home/zhanz720/library_speech/log/%x-%j.err

set -euo pipefail

module load python/3.11
module load cuda/12.9
source /home/zhanz720/envs/librispeech/bin/activate
export PYTHONUNBUFFERED=1

CODE_DIR=/home/zhanz720/library_speech
DATASET_DIR=/home/zhanz720/scratch/library_speech/dataset/LibriSpeech
CONFIG="${CONFIG:-config/experiments/bilstm_ctc.yaml}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a .pt file before sbatch}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/evaluation_${SLURM_JOB_ID}}"

cd "${CODE_DIR}"
srun python -u evaluate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --splits dev-clean test-clean
