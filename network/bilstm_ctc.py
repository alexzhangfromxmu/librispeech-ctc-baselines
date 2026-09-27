from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import CTCOutput
from .frontend import ConvSubsamplingFrontend


class ConvBiLSTMCTC(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        n_mels: int = 80,
        conv_channels: int = 128,
        hidden_size: int = 256,
        lstm_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.frontend = ConvSubsamplingFrontend(n_mels, conv_channels)
        self.encoder = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocabulary_size)

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> CTCOutput:
        x = self.frontend(features)
        output_lengths = self.frontend.output_lengths(feature_lengths)
        packed = pack_padded_sequence(
            x,
            output_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.encoder(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        logits = self.classifier(x)
        return CTCOutput(logits.log_softmax(dim=-1).transpose(0, 1), output_lengths)
