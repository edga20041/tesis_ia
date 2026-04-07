"""Helpers transversales para paths, serializacion y estructura."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Crea una carpeta si no existe y devuelve la ruta."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Escribe un diccionario como JSON."""
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return output
