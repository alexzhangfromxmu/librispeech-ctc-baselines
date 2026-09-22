from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.collate import SpeechBatch
from dataset.text import BLANK_ID, greedy_ctc_decode
from .checkpoint import (
    load_model_checkpoint,
    make_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
)
from .metrics import corpus_error_rate
from .optim import SchedulerBundle, build_optimizer, build_scheduler


def _amp_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("training.amp_dtype must be float16 or bfloat16.")


def _write_history(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _read_history(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    references: List[str] = []
    hypotheses: List[str] = []
    utterance_ids: List[str] = []
    for batch in loader:
        batch: SpeechBatch = batch.to(device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            output = model(batch.features, batch.feature_lengths)
        loss = criterion(
            output.log_probs.float(),
            batch.targets,
            output.output_lengths,
            batch.target_lengths,
        )
        total_loss += float(loss.item())
        references.extend(batch.transcripts)
        hypotheses.extend(greedy_ctc_decode(output.log_probs, output.output_lengths))
        utterance_ids.extend(batch.utterance_ids)
    return {
        "loss": total_loss / max(len(loader), 1),
        "wer": corpus_error_rate(references, hypotheses, "word"),
        "cer": corpus_error_rate(references, hypotheses, "char"),
        "references": references,
        "hypotheses": hypotheses,
        "utterance_ids": utterance_ids,
    }


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        train_generator: torch.Generator,
        config: Dict[str, Any],
        output_dir: str | Path,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.train_generator = train_generator
        self.config = config
        self.training_config = config["training"]
        self.output_dir = Path(output_dir)
        self.device = device
        self.logger = logger
        self.criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
        self.optimizer = build_optimizer(model, self.training_config)
        self.scheduler_bundle: SchedulerBundle = build_scheduler(
            self.optimizer, self.training_config, len(train_loader)
        )
        self.use_amp = bool(self.training_config.get("amp", True)) and device.type == "cuda"
        self.amp_dtype = _amp_dtype(
            str(self.training_config.get("amp_dtype", "float16"))
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

    def _resume_state(self):
        resume_path = self.training_config.get("resume_from")
        defaults = {
            "start_epoch": 1,
            "global_step": 0,
            "best_wer": float("inf"),
            "best_loss_at_wer": float("inf"),
            "best_dev_loss": float("inf"),
            "epochs_without_improvement": 0,
        }
        if not resume_path:
            return defaults
        checkpoint = restore_training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler_bundle.scheduler,
            scaler=self.scaler,
            train_generator=self.train_generator,
            path=resume_path,
            device=self.device,
        )
        defaults.update(
            start_epoch=int(checkpoint["epoch"]) + 1,
            global_step=int(checkpoint.get("global_step", 0)),
            best_wer=float(checkpoint.get("best_wer", float("inf"))),
            best_loss_at_wer=float(
                checkpoint.get("best_loss_at_wer", float("inf"))
            ),
            best_dev_loss=float(checkpoint.get("best_dev_loss", float("inf"))),
            epochs_without_improvement=int(
                checkpoint.get("epochs_without_improvement", 0)
            ),
        )
        self.logger.info(
            "Resumed checkpoint %s at epoch %d",
            resume_path,
            defaults["start_epoch"],
        )
        return defaults

    def fit(self) -> Dict[str, Any]:
        state = self._resume_state()
        history_path = self.output_dir / "history.csv"
        history = _read_history(history_path) if state["start_epoch"] > 1 else []
        best_path = self.output_dir / "checkpoints" / "best.pt"
        last_path = self.output_dir / "checkpoints" / "last.pt"
        epochs = int(self.training_config["epochs"])
        log_every = int(self.training_config.get("log_every", 0))

        self.logger.info("Model parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(
            "Batches per epoch: %d train, %d dev",
            len(self.train_loader),
            len(self.dev_loader),
        )
        self.logger.info(
            "Mixed precision: %s (%s)",
            self.use_amp,
            self.training_config.get("amp_dtype", "float16") if self.use_amp else "disabled",
        )

        for epoch in range(state["start_epoch"], epochs + 1):
            started = time.perf_counter()
            self.model.train()
            training_loss = 0.0
            for batch_index, batch in enumerate(self.train_loader, start=1):
                batch: SpeechBatch = batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    output = self.model(batch.features, batch.feature_lengths)
                loss = self.criterion(
                    output.log_probs.float(),
                    batch.targets,
                    output.output_lengths,
                    batch.target_lengths,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite CTC loss: {loss.item()}")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.training_config.get("gradient_clip", 5.0)),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler_bundle.step_per_batch:
                    self.scheduler_bundle.scheduler.step()
                training_loss += float(loss.item())
                state["global_step"] += 1
                if log_every > 0 and batch_index % log_every == 0:
                    self.logger.info(
                        "Epoch %03d batch %d/%d avg_loss=%.4f lr=%.2e",
                        epoch,
                        batch_index,
                        len(self.train_loader),
                        training_loss / batch_index,
                        self.optimizer.param_groups[0]["lr"],
                    )

            training_loss /= max(len(self.train_loader), 1)
            dev_metrics = evaluate(
                self.model,
                self.dev_loader,
                self.criterion,
                self.device,
                self.use_amp,
                self.amp_dtype,
            )
            if self.scheduler_bundle.step_on_metric:
                self.scheduler_bundle.scheduler.step(dev_metrics["loss"])

            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch,
                "train_loss": f"{training_loss:.6f}",
                "dev_loss": f"{dev_metrics['loss']:.6f}",
                "dev_wer": f"{dev_metrics['wer']:.6f}",
                "dev_cer": f"{dev_metrics['cer']:.6f}",
                "learning_rate": f"{self.optimizer.param_groups[0]['lr']:.8f}",
                "seconds": f"{elapsed:.2f}",
            }
            history.append(row)
            _write_history(history, history_path)

            wer_improved = dev_metrics["wer"] < state["best_wer"] - 1e-12
            tied_with_better_loss = (
                abs(dev_metrics["wer"] - state["best_wer"]) <= 1e-12
                and dev_metrics["loss"] < state["best_loss_at_wer"]
            )
            is_best = wer_improved or tied_with_better_loss
            if is_best:
                state["best_wer"] = dev_metrics["wer"]
                state["best_loss_at_wer"] = dev_metrics["loss"]

            if dev_metrics["loss"] < state["best_dev_loss"] - 1e-4:
                state["best_dev_loss"] = dev_metrics["loss"]
                state["epochs_without_improvement"] = 0
            else:
                state["epochs_without_improvement"] += 1

            metrics = {
                "train_loss": training_loss,
                "dev_loss": dev_metrics["loss"],
                "dev_wer": dev_metrics["wer"],
                "dev_cer": dev_metrics["cer"],
            }
            payload = make_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler_bundle.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=state["global_step"],
                best_wer=state["best_wer"],
                best_loss_at_wer=state["best_loss_at_wer"],
                best_dev_loss=state["best_dev_loss"],
                epochs_without_improvement=state["epochs_without_improvement"],
                metrics=metrics,
                config=self.config,
                train_generator=self.train_generator,
            )
            save_checkpoint(payload, last_path)
            if is_best:
                save_checkpoint(payload, best_path)

            self.logger.info(
                "Epoch %03d/%d train_loss=%.4f dev_loss=%.4f "
                "dev_WER=%.2f%% dev_CER=%.2f%% lr=%.2e time=%.1fs%s",
                epoch,
                epochs,
                training_loss,
                dev_metrics["loss"],
                dev_metrics["wer"] * 100,
                dev_metrics["cer"] * 100,
                self.optimizer.param_groups[0]["lr"],
                elapsed,
                " BEST" if is_best else "",
            )
            if state["epochs_without_improvement"] >= int(
                self.training_config["early_stopping_patience"]
            ):
                self.logger.info("Early stopping triggered.")
                break

        if not best_path.is_file():
            raise RuntimeError("Training finished without producing a best checkpoint.")
        best_checkpoint = load_model_checkpoint(self.model, best_path, self.device)
        final_metrics = evaluate(
            self.model,
            self.dev_loader,
            self.criterion,
            self.device,
            self.use_amp,
            self.amp_dtype,
        )
        self.logger.info(
            "Best checkpoint epoch=%s dev_loss=%.4f dev_WER=%.2f%% dev_CER=%.2f%%",
            best_checkpoint["epoch"],
            final_metrics["loss"],
            final_metrics["wer"] * 100,
            final_metrics["cer"] * 100,
        )
        return final_metrics
