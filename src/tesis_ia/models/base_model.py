"""Clase base para modelos del proyecto."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseModel:
    """Representacion simple de un modelo configurable."""

    family: str
    architecture: str
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Devuelve un resumen serializable del modelo."""
        return {
            "family": self.family,
            "architecture": self.architecture,
            "params": self.params,
        }
