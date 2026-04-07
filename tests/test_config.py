from tesis_ia.config import load_config, load_default_config, load_experiment_config


def test_load_default_config_contains_expected_sections():
    config = load_default_config()

    assert "project" in config
    assert "training" in config
    assert "evaluation" in config


def test_load_single_config_file():
    config = load_config("config/config.yaml")

    assert config["project"]["name"] == "tesis_ia"
    assert config["model"]["architecture"] == "dense"


def test_load_experiment_config_merges_stack():
    config = load_experiment_config("config/experiments/baseline.yaml")

    assert config["experiment"]["name"] == "baseline_dense_v1"
    assert config["training"]["epochs"] == 100
