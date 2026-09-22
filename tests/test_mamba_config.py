from pathlib import Path

from config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mamba2_experiment_configuration():
    config = load_config(PROJECT_ROOT / "config/experiments/mamba2_ctc_100epoch.yaml")
    assert config["model"]["name"] == "mamba2_ctc"
    assert config["model"]["mamba_layers"] == 10
    assert config["model"]["d_model"] == 256
    assert config["training"]["epochs"] == 100
    assert config["training"]["amp_dtype"] == "bfloat16"
