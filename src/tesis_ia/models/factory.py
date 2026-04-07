"""Factory para instanciar modelos a partir de configuracion."""

from __future__ import annotations

from typing import Any

from tesis_ia.models.base_model import BaseModel


def build_model(config: dict[str, Any]) -> BaseModel:
    """Construye un modelo base usando la seccion `model`."""
    model_config = config.get("model", {})
    family = model_config.get("family", "unknown")
    architecture = model_config.get("architecture", "undefined")

    params = {
        "hidden_layers": model_config.get("hidden_layers", []),
        "activation": model_config.get("activation"),
        "dropout_rate": model_config.get("dropout_rate"),
        "batch_normalization": model_config.get("batch_normalization", False),
    }

    return BaseModel(family=family, architecture=architecture, params=params)
