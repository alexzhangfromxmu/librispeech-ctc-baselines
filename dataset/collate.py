from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class SpeechBatch:
    features: torch.Tensor
    feature_lengths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    transcripts: List[str]
    utterance_ids: List[str]

    def to(self, device: torch.device) -> "SpeechBatch":
        return SpeechBatch(
            features=self.features.to(device, non_blocking=True),
            feature_lengths=self.feature_lengths,
            targets=self.targets.to(device, non_blocking=True),
            target_lengths=self.target_lengths,
            transcripts=self.transcripts,
            utterance_ids=self.utterance_ids,
        )


def collate_batch(batch) -> SpeechBatch:
    features, targets, transcripts, utterance_ids = zip(*batch)
    return SpeechBatch(
        features=pad_sequence(features, batch_first=True),
        feature_lengths=torch.tensor(
            [item.size(0) for item in features], dtype=torch.long
        ),
        targets=torch.cat(targets),
        target_lengths=torch.tensor(
            [item.numel() for item in targets], dtype=torch.long
        ),
        transcripts=list(transcripts),
        utterance_ids=list(utterance_ids),
    )
