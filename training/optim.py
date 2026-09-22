from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SchedulerBundle:
    scheduler: Optional[Any]
    step_per_batch: bool
    step_on_metric: bool


def build_optimizer(model: nn.Module, config: Dict[str, Any]):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_scheduler(
    optimizer, training_config: Dict[str, Any], steps_per_epoch: int
) -> SchedulerBundle:
    scheduler_config = training_config.get("scheduler", {})
    name = str(scheduler_config.get("name", "none")).lower()
    if name in ("none", "null"):
        return SchedulerBundle(None, False, False)
    if name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 2)),
            min_lr=float(scheduler_config.get("min_lr", 1e-5)),
        )
        return SchedulerBundle(scheduler, False, True)
    if name == "cosine_with_warmup":
        total_steps = max(steps_per_epoch * int(training_config["epochs"]), 1)
        warmup_steps = max(
            round(total_steps * float(scheduler_config.get("warmup_ratio", 0.05))),
            1,
        )

        def multiplier(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        return SchedulerBundle(scheduler, True, False)
    raise ValueError(f"Unknown scheduler: {name}")
