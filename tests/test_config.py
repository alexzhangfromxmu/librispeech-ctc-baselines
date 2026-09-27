from pathlib import Path

from config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_inherits_base_config():
    config = load_config(PROJECT_ROOT / "config/experiments/bilstm_ctc.yaml")
    assert config["experiment"]["seed"] == 7
    assert config["model"]["name"] == "bilstm_ctc"
    assert config["features"]["n_mels"] == 80
