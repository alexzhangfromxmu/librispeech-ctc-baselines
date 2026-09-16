# LibriSpeech CTC Baselines

This repository contains a basic end-to-end speech recognition experiment on the LibriSpeech dataset. It compares a BiLSTM encoder and a Transformer encoder under similar training conditions.

Both models were trained from scratch on the full `train-clean-100` subset using character-level CTC loss.

## Models

### BiLSTM-CTC

- Two-layer convolutional frontend
- Three-layer bidirectional LSTM
- Hidden size: 256 per direction
- Approximately 4.09 million parameters

### Transformer-CTC

- Two-layer convolutional frontend
- Five Transformer encoder layers
- Model dimension: 256
- Four attention heads
- Approximately 4.12 million parameters

Both models use 80-dimensional log-Mel features and produce character-level predictions over 29 classes, including the CTC blank token.

## Dataset

The following LibriSpeech subsets were used:

| Split | Purpose | Utterances | Duration |
|---|---|---:|---:|
| `train-clean-100` | Training | 28,539 | 100.591 hours |
| `dev-clean` | Model selection | 2,703 | 5.388 hours |
| `test-clean` | Final evaluation | 2,620 | 5.403 hours |

## Final Results

The final evaluation used FP32 inference, greedy CTC decoding, and no external language model.

| Model | Dev WER | Dev CER | Test WER | Test CER |
|---|---:|---:|---:|---:|
| BiLSTM-CTC | 34.04% | 12.63% | **33.03%** | **12.18%** |
| Transformer-CTC | 45.76% | 15.41% | 45.16% | 15.24% |

Under the current configuration, the BiLSTM model performed better than the Transformer model.

## Repository Contents

- `stage2_lstm_ctc.py`: BiLSTM-CTC training implementation
- `transformer_ctc.py`: Transformer-CTC training implementation
- `train_lstm_100h.sbatch`: Slurm job for the 100-hour BiLSTM experiment
- `train_transformer_100h.sbatch`: Slurm job for the 100-hour Transformer experiment
- `librispeech_100h_lstm_transformer_report.md`: Detailed experiment report
- `librispeech_100h_lstm_transformer_report.pdf`: PDF version of the report

## Environment

The experiments were run on an NVIDIA H100 GPU with:

- Python 3.11
- PyTorch 2.11
- torchaudio 2.11
- soundfile 0.13

## Running the Experiments

Before submitting the jobs, update the dataset, environment, and output paths in the Slurm scripts.

Submit the training jobs with:

```bash
sbatch train_lstm_100h.sbatch
sbatch train_transformer_100h.sbatch
```

The scripts expect the LibriSpeech directory to contain:

```text
LibriSpeech/
├── train-clean-100/
├── dev-clean/
└── test-clean/
```

## Notes

- The models were trained from random initialization.
- Character-level CTC targets were used.
- Decoding was performed with greedy CTC decoding.
- No beam search, pretrained acoustic model, or external language model was used.
- Model checkpoints and the LibriSpeech dataset are not included in this repository.
