from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchaudio

from config import load_config, save_config
from dataset import VOCABULARY, build_data_bundle
from network import build_model
from training.logger import create_logger
from training.reproducibility import seed_everything
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a configured LibriSpeech CTC acoustic model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def resolve_output_dir(config, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    resume_path = config["training"].get("resume_from")
    if resume_path:
        checkpoint_path = Path(resume_path).expanduser().resolve()
        if checkpoint_path.parent.name == "checkpoints":
            return checkpoint_path.parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]
    return (Path(config["output"]["runs_root"]) / run_name / f"{timestamp}_seed{seed}").resolve()


def write_environment(path: Path, device: torch.device) -> None:
    try:
        mamba_version = importlib.metadata.version("mamba-ssm")
    except importlib.metadata.PackageNotFoundError:
        mamba_version = None
    details = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mamba_ssm": mamba_version,
    }
    path.write_text(json.dumps(details, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dataset_root is not None:
        config["data"]["root"] = str(args.dataset_root.expanduser().resolve())
    if args.resume is not None:
        config["training"]["resume_from"] = str(args.resume.expanduser().resolve())

    seed = int(config["experiment"]["seed"])
    seed_everything(
        seed,
        deterministic=bool(config["experiment"].get("deterministic", True)),
        warn_only=bool(config["experiment"].get("deterministic_warn_only", True)),
    )
    device = resolve_device(args.device)
    output_dir = resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = output_dir.name
    log_path = Path(config["output"]["log_root"]).resolve() / f"{run_id}.log"
    logger = create_logger("librispeech_ctc", log_path)
    save_config(config, output_dir / "config.resolved.yaml")
    write_environment(output_dir / "environment.json", device)

    logger.info("Experiment: %s", config["experiment"]["name"])
    logger.info("Seed: %d", seed)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("Output directory: %s", output_dir)

    data = build_data_bundle(config, output_dir, device)
    model = build_model(
        config["model"],
        n_mels=int(config["features"]["n_mels"]),
        vocabulary_size=len(VOCABULARY),
    ).to(device)
    trainer = Trainer(
        model=model,
        train_loader=data.train_loader,
        dev_loader=data.dev_loader,
        train_generator=data.train_generator,
        config=config,
        output_dir=output_dir,
        device=device,
        logger=logger,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
