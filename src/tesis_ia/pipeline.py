"""Puntos de entrada internos para las tareas principales del proyecto."""

from __future__ import annotations

from typing import Any

from tesis_ia.evaluation.evaluator import Evaluator
from tesis_ia.training.trainer import Trainer


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta la corrida estructural de entrenamiento."""
    return Trainer(config).run()


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta la corrida estructural de evaluacion."""
    return Evaluator(config).run()


def run_prediction(config: dict[str, Any]) -> dict[str, Any]:
    """Devuelve un placeholder consistente para prediccion."""
    inference = config.get("inference", {})
    return {
        "status": "planned",
        "message": "El esqueleto de prediccion esta listo para implementar inferencia real.",
        "inference": {
            "return_probabilities": inference.get("return_probabilities", True),
            "threshold": inference.get("threshold", 0.5),
        },
    }
