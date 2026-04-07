from tesis_ia.config import load_config
from tesis_ia.pipeline import run_evaluation, run_prediction, run_training


def test_training_pipeline_returns_planned_status():
    payload = run_training(load_config("config/config.yaml"))

    assert payload["status"] == "planned"
    assert payload["plan"]["model"]["architecture"] == "dense"


def test_evaluation_pipeline_exposes_metrics():
    payload = run_evaluation(load_config("config/config.yaml"))

    assert payload["title"] == "evaluation_plan"
    assert "accuracy" in payload["payload"]["plan"]["metrics"]


def test_prediction_pipeline_exposes_threshold():
    payload = run_prediction(load_config("config/config.yaml"))

    assert payload["status"] == "planned"
    assert payload["inference"]["threshold"] == 0.5
