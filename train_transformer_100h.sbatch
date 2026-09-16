#!/bin/bash
#SBATCH --job-name=transformer-ctc-100h
#SBATCH --account=def-brodbeck
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhanz720/library_speech/logs/%x-%j.out
#SBATCH --error=/home/zhanz720/library_speech/logs/%x-%j.err

set -euo pipefail
module load python/3.11
source /home/zhanz720/envs/librispeech/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd /home/zhanz720/library_speech
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u transformer_ctc.py \
  --dataset-root /scratch/zhanz720/library_speech/dataset \
  --train-manifest /home/zhanz720/library_speech/stage2_outputs_100h_h100/train_manifest.tsv \
  --dev-manifest /home/zhanz720/library_speech/stage2_outputs_100h_h100/dev_manifest.tsv \
  --output-dir /home/zhanz720/library_speech/transformer_outputs_100h_h100_v2 \
  --epochs 30 \
  --batch-size 8 \
  --d-model 256 \
  --encoder-layers 5 \
  --attention-heads 4 \
  --feedforward-size 1024 \
  --dropout 0.1 \
  --learning-rate 0.001 \
  --warmup-ratio 0.05 \
  --early-stopping-patience 10 \
  --log-every 500 \
  --seed 7

echo "Finished: $(date --iso-8601=seconds)"
