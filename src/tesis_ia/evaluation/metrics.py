"""Metricas declaradas para evaluacion."""

from __future__ import annotations


def evaluation_metrics(config: dict[str, object]) -> list[str]:
    """Devuelve las metricas configuradas para evaluacion."""
    evaluation = config.get("evaluation", {})
    if isinstance(evaluation, dict):
        return list(evaluation.get("metrics", []))
    return []
