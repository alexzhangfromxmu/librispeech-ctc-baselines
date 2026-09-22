from __future__ import annotations

import re
from typing import Iterable, List

import torch


BLANK_ID = 0
VOCABULARY = ["<blank>", "'", " "] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHAR_TO_ID = {character: index for index, character in enumerate(VOCABULARY)}
ID_TO_CHAR = {index: character for index, character in enumerate(VOCABULARY)}


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z' ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_text(text: str) -> torch.Tensor:
    return torch.tensor(
        [CHAR_TO_ID[character] for character in normalize_text(text)],
        dtype=torch.long,
    )


def greedy_ctc_decode(
    log_probs: torch.Tensor, lengths: torch.Tensor
) -> List[str]:
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


def decode_ids(token_ids: Iterable[int]) -> str:
    return normalize_text("".join(ID_TO_CHAR[int(token)] for token in token_ids))
