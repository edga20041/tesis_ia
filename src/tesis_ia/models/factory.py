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
    elif family == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        max_depth = model_config.get("max_depth", 10)
        print(f"[INFO] Instanciando Arbol de Decision (max_depth={max_depth}).")
        return DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=config.get("project", {}).get("seed", 42)
        )
    elif family == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        n_estimators = model_config.get("n_estimators", 100)
        max_depth = model_config.get("max_depth", 12)
        n_jobs = model_config.get("n_jobs", -1)
        print(f"[INFO] Instanciando Random Forest (n_estimators={n_estimators}, max_depth={max_depth}).")
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=config.get("project", {}).get("seed", 42)
        )
    elif family == "lightgbm":
        import lightgbm as lgb
        n_estimators = model_config.get("n_estimators", 150)
        max_depth = model_config.get("max_depth", 8)
        learning_rate = model_config.get("learning_rate", 0.05)
        print(f"[INFO] Instanciando LightGBM (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}).")
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=config.get("project", {}).get("seed", 42),
            n_jobs=-1,
            verbose=-1
        )
    elif family == "xgboost":
        import xgboost as xgb
        n_estimators = model_config.get("n_estimators", 150)
        max_depth = model_config.get("max_depth", 6)
        learning_rate = model_config.get("learning_rate", 0.05)
        print(f"[INFO] Instanciando XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}).")
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=config.get("project", {}).get("seed", 42),
            n_jobs=-1
        )
    else:
        raise ValueError(f"Familia de modelo no soportada en esta tesis: {family}")