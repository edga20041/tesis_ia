"""Orquestador base del entrenamiento."""

from __future__ import annotations

from typing import Any

from tesis_ia.data.loader import DataLoader
from tesis_ia.data.preprocessor import Preprocessor
from tesis_ia.models.factory import build_model
from tesis_ia.training.callbacks import build_callbacks
from tesis_ia.training.metrics import training_metrics


class Trainer:
    """Coordina una corrida de entrenamiento a nivel estructural."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.model = build_model(config)

    def plan(self) -> dict[str, Any]:
        """Resume el plan de entrenamiento sin ejecutar IA real."""
        training_config = self.config.get("training", {})
        return {
            "model": self.model.summary(),
            "data": self.data_loader.describe(),
            "preprocessing": self.preprocessor.describe(),
            "training": {
                "batch_size": training_config.get("batch_size"),
                "epochs": training_config.get("epochs"),
                "learning_rate": training_config.get("learning_rate"),
                "optimizer": training_config.get("optimizer"),
                "loss_function": training_config.get("loss_function"),
                "metrics": training_metrics(self.config),
                "callbacks": build_callbacks(self.config),
            },
        }

    def run(self) -> dict[str, Any]:
        """Devuelve una corrida simulada para validar estructura."""
        return {
            "status": "planned",
            "message": "El esqueleto de entrenamiento esta listo para implementar la logica real.",
            "plan": self.plan(),
        }
