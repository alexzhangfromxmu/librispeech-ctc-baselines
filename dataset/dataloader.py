from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from training.reproducibility import seed_worker

from .collate import collate_batch
from .features import LibriSpeechDataset
from .librispeech import (
    Example,
    describe_subset,
    load_manifest,
    materialize_all_examples,
    read_transcripts,
    resolve_split,
    save_manifest,
    select_dev_examples,
    select_training_examples,
)


@dataclass
class DataBundle:
    train_loader: DataLoader
    dev_loader: DataLoader
    train_examples: List[Example]
    dev_examples: List[Example]
    train_generator: torch.Generator


def _require_dataset_root(data_config: Dict[str, Any]) -> Path:
    root = data_config.get("root")
    if root in (None, ""):
        raise ValueError(
            "Dataset root is missing. Use --dataset-root or set data.root in config."
        )
    if isinstance(root, str) and "${" in root:
        raise ValueError(f"Dataset environment variable was not resolved: {root}")
    return Path(root).expanduser()


def _prepare_examples(config: Dict[str, Any], seed: int):
    data_config = config["data"]
    train_manifest = data_config.get("train_manifest")
    dev_manifest = data_config.get("dev_manifest")
    if train_manifest or dev_manifest:
        if not train_manifest or not dev_manifest:
            raise ValueError("train_manifest and dev_manifest must be supplied together.")
        return load_manifest(train_manifest), load_manifest(dev_manifest)

    root = _require_dataset_root(data_config)
    train_split = str(data_config["train_split"])
    dev_split = str(data_config["dev_split"])
    train_index = read_transcripts(resolve_split(root, train_split))
    dev_index = read_transcripts(resolve_split(root, dev_split))
    if bool(data_config.get("full_train", False)):
        train_examples = materialize_all_examples(train_index, train_split)
    else:
        train_examples = select_training_examples(
            train_index,
            float(data_config["train_hours"]),
            seed,
            float(data_config["min_duration"]),
            float(data_config["max_duration"]),
        )
    if bool(data_config.get("full_dev", False)):
        dev_examples = materialize_all_examples(dev_index, dev_split)
    else:
        dev_examples = select_dev_examples(
            dev_index,
            int(data_config["num_dev"]),
            seed + 1,
            float(data_config["min_duration"]),
            float(data_config["max_duration"]),
        )
    return train_examples, dev_examples


def build_data_bundle(
    config: Dict[str, Any], output_dir: str | Path, device: torch.device
) -> DataBundle:
    seed = int(config["experiment"]["seed"])
    data_config = config["data"]
    train_examples, dev_examples = _prepare_examples(config, seed)
    print(describe_subset("Train subset", train_examples), flush=True)
    print(describe_subset("Dev subset", dev_examples), flush=True)
    save_manifest(train_examples, Path(output_dir) / "train_manifest.tsv")
    save_manifest(dev_examples, Path(output_dir) / "dev_manifest.tsv")

    cache_features = bool(data_config.get("cache_features", True))
    train_dataset = LibriSpeechDataset(
        train_examples, config["features"], cache_features=cache_features
    )
    dev_dataset = LibriSpeechDataset(
        dev_examples, config["features"], cache_features=cache_features
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(data_config["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 0)),
        "collate_fn": collate_batch,
        "pin_memory": bool(data_config.get("pin_memory", True))
        and device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    dev_loader = DataLoader(dev_dataset, shuffle=False, **common)
    return DataBundle(
        train_loader=train_loader,
        dev_loader=dev_loader,
        train_examples=train_examples,
        dev_examples=dev_examples,
        train_generator=generator,
    )


def build_evaluation_loader(
    examples: Sequence[Example],
    config: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    data_config = config["data"]
    dataset = LibriSpeechDataset(
        examples,
        config["features"],
        cache_features=bool(data_config.get("cache_features", True)),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        collate_fn=collate_batch,
        pin_memory=bool(data_config.get("pin_memory", True))
        and device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
