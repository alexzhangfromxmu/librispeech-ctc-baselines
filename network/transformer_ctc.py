from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import CTCOutput
from .frontend import ConvSubsamplingFrontend


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 10_000):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {inputs.size(1)} exceeds positional limit "
                f"{self.encoding.size(1)}."
            )
        return inputs + self.encoding[:, : inputs.size(1)].to(dtype=inputs.dtype)


class ConvTransformerCTC(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        n_mels: int = 80,
        conv_channels: int = 128,
        d_model: int = 256,
        encoder_layers: int = 5,
        attention_heads: int = 4,
        feedforward_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads.")
        self.frontend = ConvSubsamplingFrontend(n_mels, conv_channels)
        self.input_projection = nn.Linear(conv_channels, d_model)
        self.position_encoding = SinusoidalPositionEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(d_model, vocabulary_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> CTCOutput:
        x = self.frontend(features)
        output_lengths = self.frontend.output_lengths(feature_lengths)
        x = self.input_projection(x)
        x = self.input_dropout(self.position_encoding(x))
        time_indices = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        padding_mask = time_indices >= output_lengths.to(x.device).unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        logits = self.classifier(x)
        return CTCOutput(logits.log_softmax(dim=-1).transpose(0, 1), output_lengths)
