"""Metricas declaradas por configuracion."""

from __future__ import annotations

from typing import Any


def training_metrics(config: dict[str, Any]) -> list[str]:
    """Devuelve la lista de metricas de entrenamiento."""
    return list(config.get("training", {}).get("metrics", []))
