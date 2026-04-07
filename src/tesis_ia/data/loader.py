"""Carga de datasets y preparacion basica de metadatos."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DataLoader:
    """Localiza los directorios de datos y expone metadatos utiles."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_config = config.get("data", {})

    def describe(self) -> dict[str, Any]:
        """Devuelve un resumen simple de las rutas de datos."""
        raw_path = Path(self.data_config.get("raw_path", "data/raw"))
        processed_path = Path(self.data_config.get("processed_path", "data/processed"))
        splits_path = Path(self.data_config.get("splits_path", "data/splits"))

        return {
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "splits_path": str(splits_path),
            "validation_split": self.data_config.get("validation_split", 0.2),
            "test_split": self.data_config.get("test_split", 0.1),
        }
