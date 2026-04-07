"""Helpers para generar reportes serializables."""

from __future__ import annotations

from typing import Any


def build_report(title: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Construye un reporte simple."""
    return {
        "title": title,
        "payload": payload,
    }
