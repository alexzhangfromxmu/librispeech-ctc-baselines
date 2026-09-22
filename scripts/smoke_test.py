from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.text import BLANK_ID, VOCABULARY
from network import build_model
from training.reproducibility import seed_everything


def check_model(model_config) -> None:
    seed_everything(7, deterministic=True, warn_only=True)
    model = build_model(model_config, n_mels=80, vocabulary_size=len(VOCABULARY))
    features = torch.randn(2, 80, 80)
    feature_lengths = torch.tensor([80, 72], dtype=torch.long)
    output = model(features, feature_lengths)
    assert output.log_probs.shape[1] == 2
    assert output.log_probs.shape[2] == len(VOCABULARY)
    targets = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)(
        output.log_probs, targets, output.output_lengths, target_lengths
    )
    assert torch.isfinite(loss)


def main() -> None:
    check_model(
        {
            "name": "bilstm_ctc",
            "conv_channels": 16,
            "hidden_size": 16,
            "lstm_layers": 2,
            "dropout": 0.1,
        }
    )
    check_model(
        {
            "name": "transformer_ctc",
            "conv_channels": 16,
            "d_model": 16,
            "encoder_layers": 2,
            "attention_heads": 4,
            "feedforward_size": 32,
            "dropout": 0.1,
        }
    )
    print("Smoke test passed for BiLSTM-CTC and Transformer-CTC.")


if __name__ == "__main__":
    main()
