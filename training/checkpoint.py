from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from dataset.text import VOCABULARY
from .reproducibility import capture_rng_state, restore_rng_state


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_checkpoint(
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_wer: float,
    best_loss_at_wer: float,
    best_dev_loss: float,
    epochs_without_improvement: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    train_generator: torch.Generator,
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "train_generator_state": train_generator.get_state(),
        "best_wer": best_wer,
        "best_loss_at_wer": best_loss_at_wer,
        "best_dev_loss": best_dev_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "vocabulary": VOCABULARY,
        "model_config": config["model"],
        "feature_config": config["features"],
        "config": config,
        "metrics": metrics,
    }


def save_checkpoint(payload: Dict[str, Any], path: str | Path) -> None:
    _atomic_torch_save(payload, Path(path))


def _load_model_state(model: nn.Module, state: Dict[str, Any]) -> None:
    """Load current checkpoints and remap the old monolithic frontend keys."""
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        remapped = {}
        for key, value in state.items():
            if key.startswith("frontend.") and not key.startswith("frontend.layers."):
                key = "frontend.layers." + key[len("frontend.") :]
            remapped[key] = value
        model.load_state_dict(remapped, strict=True)


def load_model_checkpoint(
    model: nn.Module, path: str | Path, device: torch.device
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vocabulary = checkpoint.get("vocabulary")
    if vocabulary is not None and list(vocabulary) != VOCABULARY:
        raise ValueError(f"Vocabulary mismatch in checkpoint: {path}")
    _load_model_state(model, checkpoint["model_state"])
    return checkpoint


def restore_training_checkpoint(
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    train_generator: torch.Generator,
    path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = load_model_checkpoint(model, path, device)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    if checkpoint.get("train_generator_state") is not None:
        train_generator.set_state(checkpoint["train_generator_state"].cpu())
    restore_rng_state(checkpoint.get("rng_state", {}))
    return checkpoint
