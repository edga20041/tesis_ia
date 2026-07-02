from __future__ import annotations
from typing import Any
from sklearn.linear_model import LinearRegression

def build_model(config: dict[str, Any]) -> Any:
    """Construye el modelo real a partir de la configuracion."""
    model_config = config.get("model", {})
    family = model_config.get("family", "linear_model")
    
    if family == "linear_model":
        print("[INFO] Instanciando Modelo Base: Regresion Lineal Multivariada.")
        return LinearRegression()
    elif family == "lightgbm":
        # Queda preparado para la Fase 3 del cronograma
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=config.get("project", {}).get("seed", 42))
    elif family == "xgboost":
        # Queda preparado para la Fase 3 del cronograma
        import xgboost as xgb
        return xgb.XGBRegressor(random_state=config.get("project", {}).get("seed", 42))
    else:
        raise ValueError(f"Familia de modelo no soportada en esta tesis: {family}")