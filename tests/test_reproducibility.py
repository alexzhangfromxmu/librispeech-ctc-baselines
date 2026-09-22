import random

import numpy as np
import torch

from training.reproducibility import seed_everything


def sample_values():
    return random.random(), np.random.rand(), torch.rand(3)


def test_seed_everything_repeats_random_streams():
    seed_everything(3407)
    first = sample_values()
    seed_everything(3407)
    second = sample_values()
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
