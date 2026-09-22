from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import soundfile as sf

from .text import normalize_text


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


TranscriptIndex = List[Tuple[Path, str]]


def resolve_split(dataset_root: str | Path, split: str) -> Path:
    root = Path(dataset_root).expanduser()
    suffix = split.replace("-", "_")
    candidates = [
        root / split,
        root / "LibriSpeech" / split,
        root / f"LibriSpeech_{suffix}" / split,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find split '{split}'. Searched:\n  {searched}")


def read_transcripts(split_dir: str | Path) -> TranscriptIndex:
    split_path = Path(split_dir)
    examples: TranscriptIndex = []
    for transcript_file in sorted(split_path.rglob("*.trans.txt")):
        with transcript_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                utterance_id, transcript = line.rstrip().split(" ", maxsplit=1)
                audio_path = transcript_file.parent / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Missing audio file: {audio_path}")
                examples.append((audio_path.resolve(), normalize_text(transcript)))
    if not examples:
        raise RuntimeError(f"No LibriSpeech examples found under {split_path}")
    return examples


def audio_duration_seconds(audio_path: str | Path) -> float:
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
        if not min_duration <= duration <= max_duration:
            continue
        selected.append(Example(audio_path, transcript, duration))
        total_seconds += duration
        if total_seconds >= target_seconds:
            break
    if total_seconds < target_seconds:
        raise RuntimeError(
            f"Only {total_seconds / 3600:.2f} hours satisfied the filters; "
            f"requested {target_hours:.2f} hours."
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
    candidates: Sequence[Tuple[Path, str]], subset_name: str
) -> List[Example]:
    selected: List[Example] = []
    total = len(candidates)
    for index, (audio_path, transcript) in enumerate(candidates, start=1):
        selected.append(Example(audio_path, transcript, audio_duration_seconds(audio_path)))
        if index % 5000 == 0 or index == total:
            print(f"  Indexed {subset_name}: {index}/{total}", flush=True)
    return selected


def load_manifest(manifest_path: str | Path) -> List[Example]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    examples: List[Example] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"audio_path", "transcript", "duration_seconds"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        for row in reader:
            audio_path = Path(row["audio_path"])
            if not audio_path.is_file():
                raise FileNotFoundError(f"Manifest audio file not found: {audio_path}")
            examples.append(
                Example(
                    audio_path=audio_path.resolve(),
                    transcript=normalize_text(row["transcript"]),
                    duration_seconds=float(row["duration_seconds"]),
                )
            )
    if not examples:
        raise RuntimeError(f"Manifest is empty: {path}")
    return examples


def save_manifest(examples: Sequence[Example], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
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


def describe_subset(name: str, examples: Sequence[Example]) -> str:
    total_seconds = sum(example.duration_seconds for example in examples)
    speakers = len({example.speaker_id for example in examples})
    return (
        f"{name}: {len(examples)} utterances, "
        f"{total_seconds / 3600:.3f} hours, {speakers} speakers"
    )
