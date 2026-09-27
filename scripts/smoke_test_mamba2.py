from __future__ import annotations

import importlib.metadata
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The Mamba-2 smoke test requires a CUDA allocation.")
    try:
        version = importlib.metadata.version("mamba-ssm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "mamba-ssm is not installed; follow hpc/MAMBA_SETUP.md."
        ) from error

    seed_everything(7, deterministic=True, warn_only=True)
    device = torch.device("cuda")
    model = build_model(
        {
            "name": "mamba2_ctc",
            "conv_channels": 16,
            "d_model": 64,
            "mamba_layers": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "headdim": 32,
            "chunk_size": 64,
            "dropout": 0.1,
        },
        n_mels=80,
        vocabulary_size=len(VOCABULARY),
    ).to(device)
    features = torch.randn(2, 80, 80, device=device)
    feature_lengths = torch.tensor([80, 72], dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(features, feature_lengths)
    assert output.log_probs.shape[1:] == (2, len(VOCABULARY))
    assert output.output_lengths.tolist() == [20, 18]

    targets = torch.tensor(
        [3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long, device=device
    )
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)(
        output.log_probs.float(), targets, output.output_lengths, target_lengths
    )
    assert torch.isfinite(loss)
    loss.backward()
    print(
        f"Mamba-2 smoke test passed on {torch.cuda.get_device_name(0)} "
        f"with mamba-ssm {version}; loss={loss.item():.4f}."
    )


if __name__ == "__main__":
    main()
