import torch

from dataset.text import VOCABULARY, encode_text, greedy_ctc_decode, normalize_text


def test_normalization_and_encoding():
    assert normalize_text("Hello,   world!") == "HELLO WORLD"
    assert encode_text("A B").numel() == 3


def test_greedy_ctc_decode_collapses_repeats_and_blank():
    token_ids = torch.tensor([[3], [3], [0], [4]], dtype=torch.long)
    log_probs = torch.full((4, 1, len(VOCABULARY)), -100.0)
    for time, token in enumerate(token_ids[:, 0]):
        log_probs[time, 0, token] = 0.0
    assert greedy_ctc_decode(log_probs, torch.tensor([4])) == ["AB"]
