import torch

from dataset.collate import collate_batch


def test_collate_tracks_feature_and_target_lengths():
    batch = [
        (torch.ones(5, 3), torch.tensor([3, 4]), "AB", "utt-1"),
        (torch.ones(3, 3), torch.tensor([5]), "C", "utt-2"),
    ]
    result = collate_batch(batch)
    assert result.features.shape == (2, 5, 3)
    assert result.feature_lengths.tolist() == [5, 3]
    assert result.target_lengths.tolist() == [2, 1]
    assert result.utterance_ids == ["utt-1", "utt-2"]
