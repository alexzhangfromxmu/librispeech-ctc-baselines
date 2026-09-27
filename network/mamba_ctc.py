from __future__ import annotations

import torch
import torch.nn as nn

from .base import CTCOutput
from .frontend import ConvSubsamplingFrontend


def _mamba2_class():
    try:
        from mamba_ssm import Mamba2
    except ImportError as error:
        raise RuntimeError(
            "Mamba-2 requires the optional 'mamba-ssm' package. "
            "Install requirements-mamba.txt on a Linux CUDA environment."
        ) from error
    return Mamba2


class ResidualMamba2Block(nn.Module):
    """Pre-norm residual wrapper around one official Mamba-2 mixer block."""

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        chunk_size: int,
        dropout: float,
        layer_idx: int,
    ):
        super().__init__()
        Mamba2 = _mamba2_class()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.mixer(self.norm(inputs)))


class ConvMamba2CTC(nn.Module):
    """Convolutional subsampling + stacked Mamba-2 encoder + CTC head.

    Mamba-2 is causal in the acoustic time direction. Padded frames occur only
    after each utterance and are excluded from CTC by ``output_lengths``.
    """

    def __init__(
        self,
        vocabulary_size: int,
        n_mels: int = 80,
        conv_channels: int = 128,
        d_model: int = 256,
        mamba_layers: int = 10,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if mamba_layers <= 0:
            raise ValueError("mamba_layers must be positive.")
        if d_model <= 0 or expand <= 0 or headdim <= 0:
            raise ValueError("d_model, expand, and headdim must be positive.")
        if (expand * d_model) % headdim != 0:
            raise ValueError("expand * d_model must be divisible by headdim.")

        self.frontend = ConvSubsamplingFrontend(n_mels, conv_channels)
        self.input_projection = nn.Linear(conv_channels, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                ResidualMamba2Block(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    chunk_size=chunk_size,
                    dropout=dropout,
                    layer_idx=layer_index,
                )
                for layer_index in range(mamba_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocabulary_size)

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> CTCOutput:
        x = self.frontend(features)
        output_lengths = self.frontend.output_lengths(feature_lengths)
        x = self.input_dropout(self.input_projection(x))
        for block in self.encoder:
            x = block(x)
        logits = self.classifier(self.final_norm(x))
        return CTCOutput(logits.log_softmax(dim=-1).transpose(0, 1), output_lengths)
