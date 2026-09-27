# HPC usage

The Slurm output directory must exist before submitting any job:

```bash
cd /home/zhanz720/library_speech
mkdir -p log runs
```

Submit either training configuration:

```bash
sbatch hpc/train_bilstm_ctc.sbatch
sbatch hpc/train_transformer_ctc.sbatch
```

For Mamba-2, first follow `hpc/MAMBA_SETUP.md`, then run:

```bash
sbatch hpc/smoke_test_mamba2.sbatch
sbatch hpc/train_mamba2_ctc_100epoch.sbatch
```

Resume a stopped run with the normal training entry point and the same output
directory:

```bash
python train.py \
  --config config/experiments/bilstm_ctc.yaml \
  --dataset-root /home/zhanz720/scratch/library_speech/dataset/LibriSpeech \
  --output-dir runs/bilstm_ctc_100h/job_ID_seed7 \
  --resume runs/bilstm_ctc_100h/job_ID_seed7/checkpoints/last.pt \
  --device cuda
```

For evaluation, provide environment variables at submission time:

```bash
CHECKPOINT=/absolute/path/best.pt \
CONFIG=config/experiments/bilstm_ctc.yaml \
sbatch hpc/evaluate.sbatch
```
