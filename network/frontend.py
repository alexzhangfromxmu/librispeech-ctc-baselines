from __future__ import annotations

import torch
import torch.nn as nn


class ConvSubsamplingFrontend(nn.Module):
    """Two stride-2 convolutions shared by both baseline encoders."""

    def __init__(self, n_mels: int, conv_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(
                conv_channels, conv_channels, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
        )

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def forward(self, features: torch.Tensor):
        return self.layers(features.transpose(1, 2)).transpose(1, 2)
