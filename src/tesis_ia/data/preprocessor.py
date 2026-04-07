"""Preprocesamiento declarativo basado en configuracion."""

from __future__ import annotations

from typing import Any


class Preprocessor:
    """Representa las reglas de preprocesamiento configuradas."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config.get("preprocessing", {})

    def describe(self) -> dict[str, Any]:
        """Devuelve las reglas activas de preprocesamiento."""
        return dict(self.config)
