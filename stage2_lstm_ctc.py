"""Stage 2: a small LibriSpeech generalization experiment with BiLSTM + CTC.

Default experiment:
  * deterministically sample about one hour from train-clean-100
  * deterministically sample 200 utterances from dev-clean
  * train a 3-layer BiLSTM acoustic encoder for at most 30 epochs
  * save the checkpoint with the lowest development-set WER

The targets passed to CTCLoss are integer character IDs.  The network output is
a distribution over 29 mutually exclusive character classes at each time step,
which is the CTC-compatible equivalent of one-hot classification.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


BLANK_ID = 0
VOCABULARY = ["<blank>", "'", " "] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHAR_TO_ID = {char: index for index, char in enumerate(VOCABULARY)}
ID_TO_CHAR = {index: char for index, char in enumerate(VOCABULARY)}


@dataclass(frozen=True)
class Example:
    audio_path: Path
    transcript: str
    duration_seconds: float

    @property
    def utterance_id(self) -> str:
        return self.audio_path.stem

    @property
    def speaker_id(self) -> str:
        return self.utterance_id.split("-")[0]


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z' ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_text(text: str) -> torch.Tensor:
    return torch.tensor(
        [CHAR_TO_ID[character] for character in normalize_text(text)],
        dtype=torch.long,
    )


def resolve_split(dataset_root: Path, split: str) -> Path:
    suffix = split.replace("-", "_")
    candidates = [
        dataset_root / split,
        dataset_root / "LibriSpeech" / split,
        dataset_root / f"LibriSpeech_{suffix}" / split,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find split '{split}'. Searched:\n  {searched}")


def read_transcripts(split_dir: Path) -> List[Tuple[Path, str]]:
    examples: List[Tuple[Path, str]] = []
    for transcript_file in sorted(split_dir.rglob("*.trans.txt")):
        with transcript_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                utterance_id, transcript = line.rstrip().split(" ", maxsplit=1)
                audio_path = transcript_file.parent / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Missing audio file: {audio_path}")
                examples.append((audio_path, normalize_text(transcript)))
    if not examples:
        raise RuntimeError(f"No LibriSpeech examples found under {split_dir}")
    return examples


def audio_duration_seconds(audio_path: Path) -> float:
    metadata = sf.info(str(audio_path))
    return metadata.frames / metadata.samplerate


def select_training_examples(
    candidates: Sequence[Tuple[Path, str]],
    target_hours: float,
    seed: int,
    min_duration: float,
    max_duration: float,
) -> List[Example]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    target_seconds = target_hours * 3600.0
    selected: List[Example] = []
    total_seconds = 0.0
    for audio_path, transcript in shuffled:
        duration = audio_duration_seconds(audio_path)
        if duration < min_duration or duration > max_duration:
            continue
        selected.append(Example(audio_path, transcript, duration))
        total_seconds += duration
        if total_seconds >= target_seconds:
            break
    if total_seconds < target_seconds:
        raise RuntimeError(
            f"Could only select {total_seconds / 3600:.2f} hours, "
            f"less than requested {target_hours:.2f} hours."
        )
    return selected


def select_dev_examples(
    candidates: Sequence[Tuple[Path, str]],
    count: int,
    seed: int,
    min_duration: float,
    max_duration: float,
) -> List[Example]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    selected: List[Example] = []
    for audio_path, transcript in shuffled:
        duration = audio_duration_seconds(audio_path)
        if min_duration <= duration <= max_duration:
            selected.append(Example(audio_path, transcript, duration))
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Could only select {len(selected)} of {count} dev examples.")
    return selected


def materialize_all_examples(
    candidates: Sequence[Tuple[Path, str]],
    subset_name: str,
) -> List[Example]:
    """Use every utterance in an official split without duration filtering."""
    selected: List[Example] = []
    total = len(candidates)
    for index, (audio_path, transcript) in enumerate(candidates, start=1):
        selected.append(
            Example(audio_path, transcript, audio_duration_seconds(audio_path))
        )
        if index % 5000 == 0 or index == total:
            print(f"  Indexed {subset_name}: {index}/{total}")
    return selected


def save_manifest(examples: Sequence[Example], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["utterance_id", "duration_seconds", "audio_path", "transcript"])
        for example in examples:
            writer.writerow(
                [
                    example.utterance_id,
                    f"{example.duration_seconds:.3f}",
                    str(example.audio_path),
                    example.transcript,
                ]
            )


def describe_subset(name: str, examples: Sequence[Example]) -> None:
    total_seconds = sum(example.duration_seconds for example in examples)
    speakers = len({example.speaker_id for example in examples})
    print(
        f"{name}: {len(examples)} utterances, "
        f"{total_seconds / 3600:.3f} hours, {speakers} speakers"
    )


class LibriSpeechSubset(Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        n_mels: int = 80,
        cache_features: bool = True,
    ):
        self.examples = list(examples)
        self.cache_features = cache_features
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, str]] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cache_features and index in self.cache:
            return self.cache[index]

        example = self.examples[index]
        samples, sample_rate = sf.read(
            str(example.audio_path),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(samples.T.copy())
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        waveform = waveform.mean(dim=0, keepdim=True)

        features = self.to_db(self.mel(waveform)).squeeze(0).transpose(0, 1)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp_min(1e-5)
        features = (features - mean) / std
        item = (features, encode_text(example.transcript), example.transcript)
        if self.cache_features:
            self.cache[index] = item
        return item


def collate_batch(batch):
    features, targets, transcripts = zip(*batch)
    feature_lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    target_lengths = torch.tensor([item.numel() for item in targets], dtype=torch.long)
    return (
        pad_sequence(features, batch_first=True),
        feature_lengths,
        torch.cat(targets),
        target_lengths,
        list(transcripts),
    )


class ConvBiLSTMCTC(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        conv_channels: int = 128,
        hidden_size: int = 256,
        lstm_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.encoder = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, len(VOCABULARY))

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        x = self.frontend(features.transpose(1, 2)).transpose(1, 2)
        reduced_lengths = self.output_lengths(lengths)
        packed = pack_padded_sequence(
            x,
            reduced_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.encoder(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        logits = self.classifier(x)
        return logits.log_softmax(dim=-1).transpose(0, 1), reduced_lengths


def greedy_ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
    predictions = log_probs.argmax(dim=-1).transpose(0, 1).cpu()
    decoded: List[str] = []
    for sequence, length in zip(predictions, lengths.cpu()):
        characters: List[str] = []
        previous = BLANK_ID
        for token in sequence[: int(length)].tolist():
            if token != BLANK_ID and token != previous:
                characters.append(ID_TO_CHAR[token])
            previous = token
        decoded.append(normalize_text("".join(characters)))
    return decoded


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> Tuple[int, int, int]:
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    table = [[(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)]
    for row in range(1, rows):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        table[0][column] = (column, 0, 0, column)

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            substitute = table[row - 1][column - 1]
            delete = table[row - 1][column]
            insert = table[row][column - 1]
            candidates = [
                (substitute[0] + 1, substitute[1] + 1, substitute[2], substitute[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            table[row][column] = min(candidates, key=lambda item: item[0])
    _, substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions


def corpus_error_rate(references: Sequence[str], hypotheses: Sequence[str], unit: str) -> float:
    errors = total = 0
    for reference, hypothesis in zip(references, hypotheses):
        if unit == "word":
            ref_tokens, hyp_tokens = reference.split(), hypothesis.split()
        else:
            ref_tokens = list(reference.replace(" ", ""))
            hyp_tokens = list(hypothesis.replace(" ", ""))
        substitutions, deletions, insertions = edit_counts(ref_tokens, hyp_tokens)
        errors += substitutions + deletions + insertions
        total += len(ref_tokens)
    return errors / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    batches = 0
    references: List[str] = []
    hypotheses: List[str] = []
    for features, feature_lengths, targets, target_lengths, transcripts in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            log_probs, output_lengths = model(features, feature_lengths)
        loss = criterion(log_probs.float(), targets, output_lengths, target_lengths)
        total_loss += float(loss.item())
        batches += 1
        references.extend(transcripts)
        hypotheses.extend(greedy_ctc_decode(log_probs, output_lengths))
    return {
        "loss": total_loss / max(batches, 1),
        "wer": corpus_error_rate(references, hypotheses, "word"),
        "cer": corpus_error_rate(references, hypotheses, "char"),
        "references": references,
        "hypotheses": hypotheses,
    }


def write_history(history: Sequence[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def checkpoint_payload(model, optimizer, scheduler, epoch, args, metrics):
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "vocabulary": VOCABULARY,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\1workpath\library_speech\dataset"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("stage2_outputs"))
    parser.add_argument("--train-hours", type=float, default=1.0)
    parser.add_argument("--num-dev", type=int, default=200)
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Use every utterance in train-clean-100; ignores --train-hours and duration filters.",
    )
    parser.add_argument(
        "--full-dev",
        action="store_true",
        help="Use every utterance in dev-clean; ignores --num-dev and duration filters.",
    )
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Recompute log-Mel features every epoch instead of retaining them in RAM.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.full_train and args.train_hours <= 0:
        raise ValueError("--train-hours must be positive unless --full-train is used.")
    if not args.full_dev and args.num_dev <= 0:
        raise ValueError("--num-dev must be positive unless --full-dev is used.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision: {use_amp}")

    train_dir = resolve_split(args.dataset_root, "train-clean-100")
    dev_dir = resolve_split(args.dataset_root, "dev-clean")
    print("Reading transcript indexes...")
    all_train = read_transcripts(train_dir)
    all_dev = read_transcripts(dev_dir)
    print(f"Available: train={len(all_train)}, dev={len(all_dev)} utterances")

    print("Selecting deterministic subsets and reading audio metadata...")
    if args.full_train:
        train_examples = materialize_all_examples(all_train, "train-clean-100")
    else:
        train_examples = select_training_examples(
            all_train,
            args.train_hours,
            args.seed,
            args.min_duration,
            args.max_duration,
        )
    if args.full_dev:
        dev_examples = materialize_all_examples(all_dev, "dev-clean")
    else:
        dev_examples = select_dev_examples(
            all_dev,
            args.num_dev,
            args.seed + 1,
            args.min_duration,
            args.max_duration,
        )
    describe_subset("Train subset", train_examples)
    describe_subset("Dev subset", dev_examples)
    save_manifest(train_examples, args.output_dir / "train_manifest.tsv")
    save_manifest(dev_examples, args.output_dir / "dev_manifest.tsv")

    cache_features = not args.no_feature_cache
    print(f"Feature cache in RAM: {cache_features}")
    train_dataset = LibriSpeechSubset(train_examples, cache_features=cache_features)
    dev_dataset = LibriSpeechSubset(dev_examples, cache_features=cache_features)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "collate_fn": collate_batch,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    model = ConvBiLSTMCTC(
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")
    print(f"Batches per epoch: {len(train_loader)} train, {len(dev_loader)} dev")

    best_wer = float("inf")
    best_loss_at_wer = float("inf")
    best_dev_loss = float("inf")
    epochs_without_loss_improvement = 0
    history: List[dict] = []
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        training_loss = 0.0
        for features, feature_lengths, targets, target_lengths, _ in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                log_probs, output_lengths = model(features, feature_lengths)
            loss = criterion(log_probs.float(), targets, output_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite CTC loss encountered: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            training_loss += float(loss.item())

        training_loss /= max(len(train_loader), 1)
        dev_metrics = evaluate(model, dev_loader, criterion, device, use_amp)
        scheduler.step(dev_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": f"{training_loss:.6f}",
            "dev_loss": f"{dev_metrics['loss']:.6f}",
            "dev_wer": f"{dev_metrics['wer']:.6f}",
            "dev_cer": f"{dev_metrics['cer']:.6f}",
            "learning_rate": f"{current_lr:.8f}",
            "seconds": f"{elapsed:.2f}",
        }
        history.append(row)
        write_history(history, args.output_dir / "history.csv")

        metrics_for_checkpoint = {
            "train_loss": training_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_wer": dev_metrics["wer"],
            "dev_cer": dev_metrics["cer"],
        }
        payload = checkpoint_payload(
            model, optimizer, scheduler, epoch, args, metrics_for_checkpoint
        )
        torch.save(payload, last_path)

        wer_improved = dev_metrics["wer"] < best_wer - 1e-12
        wer_tied_with_better_loss = (
            abs(dev_metrics["wer"] - best_wer) <= 1e-12
            and dev_metrics["loss"] < best_loss_at_wer
        )
        if wer_improved or wer_tied_with_better_loss:
            best_wer = dev_metrics["wer"]
            best_loss_at_wer = dev_metrics["loss"]
            torch.save(payload, best_path)
            best_marker = " BEST"
        else:
            best_marker = ""

        if dev_metrics["loss"] < best_dev_loss - 1e-4:
            best_dev_loss = dev_metrics["loss"]
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={training_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_WER={dev_metrics['wer'] * 100:.2f}% "
            f"dev_CER={dev_metrics['cer'] * 100:.2f}% "
            f"lr={current_lr:.2e} time={elapsed:.1f}s{best_marker}"
        )

        if epochs_without_loss_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping: dev loss did not improve for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    final_metrics = evaluate(model, dev_loader, criterion, device, use_amp)
    print("\nBest checkpoint summary")
    print(f"Epoch:    {best_checkpoint['epoch']}")
    print(f"Dev loss: {final_metrics['loss']:.4f}")
    print(f"Dev WER:  {final_metrics['wer'] * 100:.2f}%")
    print(f"Dev CER:  {final_metrics['cer'] * 100:.2f}%")
    print(f"Best model: {best_path}")
    print(f"History:    {args.output_dir / 'history.csv'}")
    print("\nSample dev predictions:")
    for reference, hypothesis in list(
        zip(final_metrics["references"], final_metrics["hypotheses"])
    )[:5]:
        print(f"REF: {reference}")
        print(f"HYP: {hypothesis or '<empty>'}")
        print()


if __name__ == "__main__":
    main()
