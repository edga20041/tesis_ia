"""CLI reusable para los comandos principales del proyecto."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from tesis_ia.config import load_config
from tesis_ia.data.loader import DataLoader
from tesis_ia.data.preprocessor import Preprocessor
from tesis_ia.logger import configure_logging
from tesis_ia.pipeline import run_evaluation, run_prediction, run_training


Runner = Callable[[dict[str, Any]], dict[str, Any]]


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config/config.yaml", help="Ruta al archivo de configuracion.")
    return parser


def _configure_runtime(config: dict[str, Any]) -> None:
    logging_config = config.get("logging", {})
    configure_logging(
        level=logging_config.get("level", "INFO"),
        fmt=logging_config.get("format"),
        file_path=logging_config.get("file"),
    )


def _run(description: str, runner: Runner) -> None:
    parser = _base_parser(description)
    args = parser.parse_args()
    config = load_config(args.config)
    _configure_runtime(config)
    print(json.dumps(runner(config), indent=2, ensure_ascii=True))


def train_main() -> None:
    _run("Planifica una corrida de entrenamiento.", run_training)


def evaluate_main() -> None:
    _run("Planifica una corrida de evaluacion.", run_evaluation)


def predict_main() -> None:
    _run("Planifica una corrida de prediccion.", run_prediction)


def data_main() -> None:
    parser = _base_parser("Resume la preparacion de datos del proyecto.")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = {
        "status": "planned",
        "data": DataLoader(config).describe(),
        "preprocessing": Preprocessor(config).describe(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
