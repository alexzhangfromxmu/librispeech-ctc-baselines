# Mamba-2 server setup

The Mamba experiment uses the official `mamba-ssm` package on Linux/CUDA. It
is intentionally kept out of the baseline requirements so the BiLSTM and
Transformer environments continue to work without optional CUDA packages.

Activate the same environment used by the other experiments:

```bash
module load python/3.11
source /home/zhanz720/envs/librispeech/bin/activate
```

Install the pinned Mamba dependency after PyTorch is already available:

```bash
python -m pip install --upgrade packaging ninja wheel
python -m pip install "mamba-ssm[causal-conv1d]==2.3.2.post1" --no-build-isolation
```

`--no-build-isolation` is required by the upstream project so installation
uses the environment's existing CUDA-enabled PyTorch rather than an isolated
CPU-only PyTorch build.

Check the import and installed version:

```bash
python -c "import importlib.metadata as m; from mamba_ssm import Mamba2; print(m.version('mamba-ssm')); print('Mamba-2 import passed')"
```

The Mamba kernels require a CUDA node. Submit the dedicated smoke test rather
than trying to execute the model on a login node:

```bash
sbatch hpc/smoke_test_mamba2.sbatch
```

After the smoke test passes, submit training:

```bash
sbatch hpc/train_mamba2_ctc_100epoch.sbatch
```
