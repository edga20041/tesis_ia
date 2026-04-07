"""Resolucion de rutas importantes del proyecto."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_project_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Normaliza las rutas principales del proyecto."""
    root = Path(__file__).resolve().parents[3]
    raw_paths = config.get("paths", {})

    resolved = {
        "root": root,
        "data_dir": root / raw_paths.get("data_dir", "data"),
        "artifacts_dir": root / raw_paths.get("artifacts_dir", "artifacts"),
        "outputs_dir": root / raw_paths.get("outputs_dir", "outputs"),
        "logs_dir": root / raw_paths.get("logs_dir", "logs"),
    }
    return resolved
