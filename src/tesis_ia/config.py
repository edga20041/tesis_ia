"""Helpers para cargar y combinar configuraciones YAML del proyecto."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_FILES = (
    "base.yaml",
    "data.yaml",
    "model.yaml",
    "training.yaml",
    "evaluation.yaml",
)


def project_root() -> Path:
    """Devuelve la raiz del proyecto."""
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Devuelve la carpeta de configuraciones."""
    return project_root() / "config"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Carga un archivo YAML y devuelve un diccionario."""
    yaml_path = Path(path)
    if not yaml_path.is_absolute():
        yaml_path = project_root() / yaml_path

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"La configuracion en {yaml_path} debe ser un diccionario.")

    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusiona diccionarios de forma recursiva."""
    result = deepcopy(base)

    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = deep_merge(current, value)
        else:
            result[key] = deepcopy(value)

    return result


def load_default_config() -> dict[str, Any]:
    """Carga y combina la configuracion recomendada por dominio."""
    merged: dict[str, Any] = {}
    for file_name in DEFAULT_CONFIG_FILES:
        merged = deep_merge(merged, load_yaml(config_dir() / file_name))
    return merged


def load_experiment_config(experiment_file: str | Path) -> dict[str, Any]:
    experiment = load_yaml(experiment_file)
    stack = experiment.get("config_stack", [])

    merged: dict[str, Any] = {}
    for item in stack:
        merged = deep_merge(merged, load_yaml(item))

    return deep_merge(merged, {"experiment": experiment.get("experiment", {})})


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return load_default_config()

    config_path = Path(path)
    if config_path.name.endswith(".yaml") and "experiments" in config_path.parts:
        return load_experiment_config(config_path)

    return load_yaml(config_path)
