"""Configuracion basica de logging para el proyecto."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", fmt: str | None = None, file_path: str | None = None) -> None:
    """Configura logging a consola y opcionalmente a archivo."""
    log_format = fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=log_format, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger con el nombre indicado."""
    return logging.getLogger(name)
