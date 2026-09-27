from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from .librispeech import Example
from .text import encode_text


class LogMelFeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.sample_rate = int(config["sample_rate"])
        self.normalize = bool(config.get("normalize", True))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(config["n_fft"]),
            win_length=int(config["win_length"]),
            hop_length=int(config["hop_length"]),
            n_mels=int(config["n_mels"]),
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, audio_path) -> torch.Tensor:
        samples, sample_rate = sf.read(
            str(audio_path), dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(samples.T.copy())
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        waveform = waveform.mean(dim=0, keepdim=True)
        features = self.to_db(self.mel(waveform)).squeeze(0).transpose(0, 1)
        if self.normalize:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True).clamp_min(1e-5)
            features = (features - mean) / std
        return features


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        feature_config: Dict[str, Any],
        cache_features: bool = True,
    ):
        self.examples = list(examples)
        self.extract_features = LogMelFeatureExtractor(feature_config)
        self.cache_features = cache_features
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, str, str]] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        if self.cache_features and index in self.cache:
            return self.cache[index]
        example = self.examples[index]
        item = (
            self.extract_features(example.audio_path),
            encode_text(example.transcript),
            example.transcript,
            example.utterance_id,
        )
        if self.cache_features:
            self.cache[index] = item
        return item
