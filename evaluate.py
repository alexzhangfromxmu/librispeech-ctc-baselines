from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from config import load_config
from dataset import (
    BLANK_ID,
    VOCABULARY,
    build_evaluation_loader,
    materialize_all_examples,
    read_transcripts,
    resolve_split,
)
from network import build_model
from training.checkpoint import load_model_checkpoint
from training.trainer import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one LibriSpeech CTC checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["dev-clean", "test-clean"]
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def write_predictions(path: Path, metrics) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["utterance_id", "reference", "hypothesis"])
        writer.writerows(
            zip(
                metrics["utterance_ids"],
                metrics["references"],
                metrics["hypotheses"],
            )
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"]["root"] = str(args.dataset_root.expanduser().resolve())
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"],
        n_mels=int(config["features"]["n_mels"]),
        vocabulary_size=len(VOCABULARY),
    ).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    results = {}
    for split in args.splits:
        examples = materialize_all_examples(
            read_transcripts(resolve_split(args.dataset_root, split)), split
        )
        loader = build_evaluation_loader(examples, config, device)
        metrics = evaluate(
            model,
            loader,
            criterion,
            device,
            use_amp=False,
            amp_dtype=torch.float32,
        )
        write_predictions(output_dir / f"{split}_predictions.tsv", metrics)
        results[split] = {
            "checkpoint_epoch": checkpoint.get("epoch"),
            "num_utterances": len(metrics["references"]),
            "loss": metrics["loss"],
            "wer": metrics["wer"],
            "cer": metrics["cer"],
        }
        print(
            f"{split}: loss={metrics['loss']:.4f} "
            f"WER={metrics['wer'] * 100:.2f}% CER={metrics['cer'] * 100:.2f}%"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
