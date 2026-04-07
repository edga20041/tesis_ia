"""Callbacks declarativos para el pipeline de entrenamiento."""

from __future__ import annotations

from typing import Any


def build_callbacks(config: dict[str, Any]) -> dict[str, Any]:
    """Devuelve la configuracion de callbacks del entrenamiento."""
    training_config = config.get("training", {})
    return {
        "early_stopping": training_config.get("early_stopping", {}),
        "checkpoint": training_config.get("checkpoint", {}),
    }
