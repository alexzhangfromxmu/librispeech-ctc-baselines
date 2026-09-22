from pathlib import Path

import torch

from dataset.text import VOCABULARY
from network import build_model
from training.checkpoint import load_model_checkpoint, save_checkpoint


def test_model_checkpoint_round_trip(tmp_path: Path):
    model_config = {
        "name": "bilstm_ctc",
        "conv_channels": 8,
        "hidden_size": 8,
        "lstm_layers": 1,
        "dropout": 0.0,
    }
    source = build_model(model_config, 80, len(VOCABULARY))
    path = tmp_path / "model.pt"
    save_checkpoint(
        {"model_state": source.state_dict(), "vocabulary": VOCABULARY}, path
    )
    restored = build_model(model_config, 80, len(VOCABULARY))
    load_model_checkpoint(restored, path, torch.device("cpu"))
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
