import pytest
import torch

from dataset.text import VOCABULARY
from network import build_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 requires CUDA")
def test_mamba2_model_contract_on_cuda():
    pytest.importorskip("mamba_ssm")
    model = build_model(
        {
            "name": "mamba2_ctc",
            "conv_channels": 16,
            "d_model": 64,
            "mamba_layers": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "headdim": 32,
            "chunk_size": 64,
            "dropout": 0.1,
        },
        n_mels=80,
        vocabulary_size=len(VOCABULARY),
    ).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            torch.randn(2, 64, 80, device="cuda"), torch.tensor([64, 60])
        )
    assert output.log_probs.shape[1:] == (2, len(VOCABULARY))
    assert output.output_lengths.tolist() == [16, 15]
