import pytest
import torch

from dataset.text import VOCABULARY
from network import build_model


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "name": "bilstm_ctc",
            "conv_channels": 16,
            "hidden_size": 16,
            "lstm_layers": 2,
            "dropout": 0.1,
        },
        {
            "name": "transformer_ctc",
            "conv_channels": 16,
            "d_model": 16,
            "encoder_layers": 2,
            "attention_heads": 4,
            "feedforward_size": 32,
            "dropout": 0.1,
        },
    ],
)
def test_model_contract(model_config):
    model = build_model(model_config, n_mels=80, vocabulary_size=len(VOCABULARY))
    output = model(torch.randn(2, 64, 80), torch.tensor([64, 60]))
    assert output.log_probs.shape[1:] == (2, len(VOCABULARY))
    assert output.output_lengths.tolist() == [16, 15]
