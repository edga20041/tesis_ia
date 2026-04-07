"""Evaluador base para el proyecto."""

from __future__ import annotations

from typing import Any

from tesis_ia.evaluation.metrics import evaluation_metrics
from tesis_ia.evaluation.reporting import build_report
from tesis_ia.evaluation.visualization import requested_plots


class Evaluator:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def plan(self) -> dict[str, Any]:
        """Devuelve el plan de evaluacion."""
        evaluation = self.config.get("evaluation", {})
        return {
            "batch_size": evaluation.get("batch_size"),
            "metrics": evaluation_metrics(self.config),
            "plots": requested_plots(self.config),
        }

    def run(self) -> dict[str, Any]:
        """Devuelve una respuesta estructural de evaluacion."""
        return build_report(
            title="evaluation_plan",
            payload={
                "status": "planned",
                "message": "El esqueleto de evaluacion esta listo para conectar resultados reales.",
                "plan": self.plan(),
            },
        )
