"""Unified final evaluation for the LibriSpeech LSTM-CTC and Transformer-CTC.

The script evaluates both selected checkpoints with exactly the same examples,
features, text normalization, greedy CTC decoder, and WER/CER implementation.
It first evaluates dev-clean (a reproducibility check) and then test-clean (the
final held-out result).  No language model or beam search is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import stage2_lstm_ctc as lstm_code
import transformer_ctc as transformer_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lstm-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev-clean", "test-clean"],
        default=["dev-clean", "test-clean"],
        help="Evaluate dev-clean first to reproduce prior results, then test-clean.",
    )
    return parser.parse_args()


def checkpoint_arg(checkpoint: dict, name: str, default):
    return checkpoint.get("args", {}).get(name, default)


def load_lstm(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    verify_vocabulary(checkpoint, checkpoint_path)
    model = lstm_code.ConvBiLSTMCTC(
        hidden_size=int(checkpoint_arg(checkpoint, "hidden_size", 256)),
        lstm_layers=int(checkpoint_arg(checkpoint, "lstm_layers", 3)),
        dropout=float(checkpoint_arg(checkpoint, "dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def load_transformer(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    verify_vocabulary(checkpoint, checkpoint_path)
    model = transformer_code.ConvTransformerCTC(
        d_model=int(checkpoint_arg(checkpoint, "d_model", 256)),
        encoder_layers=int(checkpoint_arg(checkpoint, "encoder_layers", 5)),
        attention_heads=int(checkpoint_arg(checkpoint, "attention_heads", 4)),
        feedforward_size=int(checkpoint_arg(checkpoint, "feedforward_size", 1024)),
        dropout=float(checkpoint_arg(checkpoint, "dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def verify_vocabulary(checkpoint: dict, path: Path) -> None:
    vocabulary = checkpoint.get("vocabulary")
    if vocabulary is not None and list(vocabulary) != lstm_code.VOCABULARY:
        raise ValueError(f"Vocabulary mismatch in checkpoint: {path}")


def write_predictions(
    path: Path,
    utterance_ids: List[str],
    references: List[str],
    hypotheses: List[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["utterance_id", "reference", "hypothesis"])
        writer.writerows(zip(utterance_ids, references, hypotheses))


@torch.inference_mode()
def evaluate_model(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    utterance_ids: List[str],
    device: torch.device,
    log_every: int,
) -> Dict:
    criterion = nn.CTCLoss(blank=lstm_code.BLANK_ID, zero_infinity=True)
    total_loss = 0.0
    references: List[str] = []
    hypotheses: List[str] = []
    started = time.perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        features, feature_lengths, targets, target_lengths, transcripts = batch
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # FP32 is deliberately used for both architectures for a fair final test.
        log_probs, output_lengths = model(features, feature_lengths)
        loss = criterion(log_probs.float(), targets, output_lengths, target_lengths)
        total_loss += float(loss.item())
        references.extend(transcripts)
        hypotheses.extend(lstm_code.greedy_ctc_decode(log_probs, output_lengths))

        if log_every > 0 and (
            batch_index % log_every == 0 or batch_index == len(loader)
        ):
            print(
                f"  {name}: batch {batch_index}/{len(loader)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if len(references) != len(utterance_ids):
        raise RuntimeError("Prediction count does not match the evaluation manifest.")

    return {
        "loss": total_loss / max(len(loader), 1),
        "wer": lstm_code.corpus_error_rate(references, hypotheses, "word"),
        "cer": lstm_code.corpus_error_rate(references, hypotheses, "char"),
        "seconds": time.perf_counter() - started,
        "num_utterances": len(references),
        "references": references,
        "hypotheses": hypotheses,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Final evaluation requires a Slurm GPU allocation.")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Inference precision: FP32 for both models")

    print(f"Loading LSTM checkpoint: {args.lstm_checkpoint}")
    lstm_model, lstm_checkpoint = load_lstm(args.lstm_checkpoint, device)
    print(
        f"  selected epoch={lstm_checkpoint.get('epoch', 'unknown')}, "
        f"parameters={sum(p.numel() for p in lstm_model.parameters()):,}"
    )
    print(f"Loading Transformer checkpoint: {args.transformer_checkpoint}")
    transformer_model, transformer_checkpoint = load_transformer(
        args.transformer_checkpoint, device
    )
    print(
        f"  selected epoch={transformer_checkpoint.get('epoch', 'unknown')}, "
        f"parameters={sum(p.numel() for p in transformer_model.parameters()):,}"
    )

    models = [("lstm", lstm_model), ("transformer", transformer_model)]
    summary_rows: List[Dict] = []
    json_results: Dict[str, Dict] = {}

    for split in args.splits:
        print(f"\nPreparing full {split} ...", flush=True)
        split_dir = lstm_code.resolve_split(args.dataset_root, split)
        transcript_index = lstm_code.read_transcripts(split_dir)
        examples = lstm_code.materialize_all_examples(transcript_index, split)
        lstm_code.describe_subset(split, examples)
        lstm_code.save_manifest(examples, args.output_dir / f"{split}_manifest.tsv")

        # One dataset/cache is shared by both models, guaranteeing identical input.
        dataset = lstm_code.LibriSpeechSubset(examples, cache_features=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lstm_code.collate_batch,
            pin_memory=True,
        )
        utterance_ids = [example.utterance_id for example in examples]
        json_results[split] = {}

        for model_name, model in models:
            print(f"Evaluating {model_name} on {split} ...", flush=True)
            metrics = evaluate_model(
                model_name,
                model,
                loader,
                utterance_ids,
                device,
                args.log_every,
            )
            print(
                f"{model_name} {split}: loss={metrics['loss']:.4f} "
                f"WER={metrics['wer'] * 100:.2f}% "
                f"CER={metrics['cer'] * 100:.2f}% "
                f"time={metrics['seconds']:.1f}s",
                flush=True,
            )
            write_predictions(
                args.output_dir / f"{model_name}_{split}_predictions.tsv",
                utterance_ids,
                metrics["references"],
                metrics["hypotheses"],
            )
            summary = {
                "model": model_name,
                "split": split,
                "num_utterances": metrics["num_utterances"],
                "loss": metrics["loss"],
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "seconds": metrics["seconds"],
            }
            summary_rows.append(summary)
            json_results[split][model_name] = summary

            print("Sample predictions:")
            for reference, hypothesis in list(
                zip(metrics["references"], metrics["hypotheses"])
            )[:3]:
                print(f"REF: {reference}")
                print(f"HYP: {hypothesis or '<empty>'}")
                print()

        # Release the split-level feature cache before preparing the next split.
        del loader, dataset, examples, transcript_index
        gc.collect()

    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_results, handle, indent=2)

    print("\nFinal comparison")
    print("model        split       utterances   WER       CER")
    for row in summary_rows:
        print(
            f"{row['model']:<12} {row['split']:<11} "
            f"{row['num_utterances']:>10}   "
            f"{row['wer'] * 100:>6.2f}%   {row['cer'] * 100:>6.2f}%"
        )
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
