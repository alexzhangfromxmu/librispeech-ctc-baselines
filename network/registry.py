from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .bilstm_ctc import ConvBiLSTMCTC
from .transformer_ctc import ConvTransformerCTC


def build_model(
    model_config: Dict[str, Any], n_mels: int, vocabulary_size: int
) -> nn.Module:
    name = str(model_config["name"]).lower()
    common = {
        "vocabulary_size": vocabulary_size,
        "n_mels": n_mels,
        "conv_channels": int(model_config.get("conv_channels", 128)),
    }
    if name == "bilstm_ctc":
        return ConvBiLSTMCTC(
            **common,
            hidden_size=int(model_config.get("hidden_size", 256)),
            lstm_layers=int(model_config.get("lstm_layers", 3)),
            dropout=float(model_config.get("dropout", 0.2)),
        )
    if name == "transformer_ctc":
        return ConvTransformerCTC(
            **common,
            d_model=int(model_config.get("d_model", 256)),
            encoder_layers=int(model_config.get("encoder_layers", 5)),
            attention_heads=int(model_config.get("attention_heads", 4)),
            feedforward_size=int(model_config.get("feedforward_size", 1024)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if name in ("mamba_ctc", "mamba2_ctc"):
        from .mamba_ctc import ConvMamba2CTC

        return ConvMamba2CTC(
            **common,
            d_model=int(model_config.get("d_model", 256)),
            mamba_layers=int(model_config.get("mamba_layers", 10)),
            d_state=int(model_config.get("d_state", 64)),
            d_conv=int(model_config.get("d_conv", 4)),
            expand=int(model_config.get("expand", 2)),
            headdim=int(model_config.get("headdim", 64)),
            chunk_size=int(model_config.get("chunk_size", 256)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    supported = "bilstm_ctc, transformer_ctc, mamba2_ctc"
    raise ValueError(f"Unknown model '{name}'. Supported models: {supported}")
