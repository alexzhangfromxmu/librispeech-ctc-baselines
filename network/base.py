from __future__ import annotations

from typing import NamedTuple

import torch


class CTCOutput(NamedTuple):
    log_probs: torch.Tensor
    output_lengths: torch.Tensor
