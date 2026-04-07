"""Descripciones de visualizaciones a generar."""

from __future__ import annotations


def requested_plots(config: dict[str, object]) -> list[str]:
    """Devuelve las visualizaciones pedidas en config."""
    evaluation = config.get("evaluation", {})
    if isinstance(evaluation, dict):
        return list(evaluation.get("plot_types", []))
    return []
