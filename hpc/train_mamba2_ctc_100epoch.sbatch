#!/bin/bash
#SBATCH --job-name=mamba2-ctc-100e
#SBATCH --account=def-brodbeck
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhanz720/library_speech/log/%x-%j.out
#SBATCH --error=/home/zhanz720/library_speech/log/%x-%j.err

set -euo pipefail

module load python/3.11
source /home/zhanz720/envs/librispeech/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CODE_DIR=/home/zhanz720/library_speech
DATASET_DIR=/home/zhanz720/scratch/library_speech/dataset/LibriSpeech
OUTPUT_DIR="${CODE_DIR}/runs/mamba2_ctc_100h/job_${SLURM_JOB_ID}_seed7"

cd "${CODE_DIR}"
python -c "from mamba_ssm import Mamba2; print('Mamba-2 import passed')"
srun python -u train.py \
  --config config/experiments/mamba2_ctc_100epoch.yaml \
  --dataset-root "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda
