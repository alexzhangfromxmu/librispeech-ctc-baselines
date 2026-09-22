# Modular LibriSpeech CTC training

This project is the structured replacement for the previous monolithic
LibriSpeech scripts. It provides character-level BiLSTM-CTC,
Transformer-CTC, and Mamba-2-CTC architectures while sharing one data pipeline, trainer,
evaluation path, configuration system, and reproducibility policy.

## Directory responsibilities

- `network/`: acoustic models and the model registry.
- `dataset/`: LibriSpeech indexing, feature extraction, text encoding, and batching.
- `training/`: training loop, metrics, optimizer, checkpoint, logging, and seeds.
- `config/`: common settings and model-specific experiment settings.
- `hpc/`: Slurm submission scripts for the server.
- `log/`: text and Slurm logs; generated files are not source code.
- `runs/`: manifests, resolved configs, metrics, and checkpoints.
- `tests/`: fast tests that do not require LibriSpeech.

The dataset remains outside this repository.

## Local setup

Create an environment and install dependencies. PyTorch and torchaudio must be
installed as a matching pair for the local CUDA or CPU environment.

```powershell
python -m pip install -r requirements.txt
```

Run the fast structural test:

```powershell
python scripts/smoke_test.py
python -m pytest
```

Example local training commands:

```powershell
$env:PYTHONHASHSEED = "7"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python train.py --config config/experiments/bilstm_ctc.yaml --dataset-root D:\1workpath\library_speech\dataset
```

```powershell
python train.py --config config/experiments/transformer_ctc.yaml --dataset-root D:\1workpath\library_speech\dataset
```

For a quick local experiment, copy an experiment YAML and change
`full_train`, `full_dev`, `train_hours`, `num_dev`, and `epochs`.

## Server location

Upload the contents of this directory into:

```text
/home/zhanz720/library_speech
```

The configured server dataset path is:

```text
/home/zhanz720/scratch/library_speech/dataset/LibriSpeech
```

Before the first submission:

```bash
cd /home/zhanz720/library_speech
mkdir -p log runs
python scripts/smoke_test.py
```

Then submit:

```bash
sbatch hpc/train_bilstm_ctc.sbatch
sbatch hpc/train_transformer_ctc.sbatch
```

Mamba-2 is an optional Linux/CUDA experiment. Follow `hpc/MAMBA_SETUP.md`, run
the dedicated GPU smoke test, and then submit:

```bash
sbatch hpc/smoke_test_mamba2.sbatch
sbatch hpc/train_mamba2_ctc_100epoch.sbatch
```

Both jobs call the same `train.py`. The selected YAML file is the only
model-specific entry point.

## Reproducibility

The configured seed is applied to Python, NumPy, PyTorch CPU/CUDA, DataLoader
workers, and the DataLoader shuffle generator. Deterministic PyTorch behavior
is enabled, warnings are surfaced for unsupported operations, and cuDNN
benchmarking is disabled. Set `deterministic_warn_only: false` to make any
unsupported nondeterministic operation fail immediately. Checkpoints include RNG states,
the DataLoader generator state, optimizer, scheduler, gradient scaler, resolved
configuration, vocabulary, epoch, and global step.

For best reproducibility, use the same hardware, CUDA, PyTorch, torchaudio, and
configuration. Cross-version or cross-hardware results are not guaranteed to
be bitwise identical.

## Run contents

Each run contains:

```text
run_directory/
├── config.resolved.yaml
├── environment.json
├── train_manifest.tsv
├── dev_manifest.tsv
├── history.csv
└── checkpoints/
    ├── best.pt
    └── last.pt
```

Resume at an epoch boundary by using the same output directory:

```powershell
python train.py --config config/experiments/bilstm_ctc.yaml `
  --dataset-root D:\1workpath\library_speech\dataset `
  --output-dir runs\existing_run `
  --resume runs\existing_run\checkpoints\last.pt
```

## Adding another model

1. Add one implementation under `network/` that returns `CTCOutput`.
2. Register its name in `network/registry.py`.
3. Add an experiment YAML under `config/experiments/`.

No dataset or trainer changes should be needed.

## Mamba-2 encoder

`network/mamba_ctc.py` uses the official `mamba_ssm.Mamba2` block. The default
experiment keeps the common convolutional frontend and CTC head, then replaces
the recurrent/attention encoder with ten residual Mamba-2 blocks. Mamba is an
optional dependency because the official optimized implementation targets
Linux with CUDA; the existing BiLSTM and Transformer remain usable without it.

