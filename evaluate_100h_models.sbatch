#!/bin/bash
#SBATCH --job-name=ctc-final-eval
#SBATCH --account=def-brodbeck
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhanz720/library_speech/logs/%x-%j.out
#SBATCH --error=/home/zhanz720/library_speech/logs/%x-%j.err

set -euo pipefail

module load python/3.11
source /home/zhanz720/envs/librispeech/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

CODE_DIR=/home/zhanz720/library_speech
DATASET_DIR=/scratch/zhanz720/library_speech/dataset
OUTPUT_DIR=/home/zhanz720/library_speech/final_evaluation_100h
LSTM_CHECKPOINT=/home/zhanz720/library_speech/stage2_outputs_100h_h100/best_model.pt
TRANSFORMER_CHECKPOINT=/home/zhanz720/library_speech/transformer_outputs_100h_h100_v2/best_model.pt

cd "${CODE_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Python: $(command -v python)"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

test -f "${LSTM_CHECKPOINT}"
test -f "${TRANSFORMER_CHECKPOINT}"
test -d "${DATASET_DIR}/LibriSpeech/dev-clean"
test -d "${DATASET_DIR}/LibriSpeech/test-clean"

srun python -u evaluate_ctc_models.py \
  --dataset-root "${DATASET_DIR}" \
  --lstm-checkpoint "${LSTM_CHECKPOINT}" \
  --transformer-checkpoint "${TRANSFORMER_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
  --num-workers 0 \
  --log-every 100 \
  --splits dev-clean test-clean

echo "Finished: $(date --iso-8601=seconds)"
