"""Script del Torneo de Modelos (Champion/Challenger).

Entrena y evalúa en igualdad de condiciones:
1. Baseline: Regresión Lineal
2. Challenger 1: Árbol de Decisión (Decision Tree)
3. Challenger 2: Random Forest (Bosque Aleatorio)
4. Challenger 3: LightGBM (Gradient Boosting de Histograma)
5. Challenger 4: XGBoost (Extreme Gradient Boosting)

Calcula MAE, RMSE, MAPE, R2 y extrae la Importancia de Variables (Feature Importance).
"""

from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tesis_ia.data.preprocessor import DataPreprocessor
from tesis_ia.models.factory import build_model


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula las metricas tecnicas y de negocio evitando divisiones por cero."""
    # Evitar predicciones negativas en retail (las ventas no pueden ser < 0)
    y_pred_clean = np.clip(y_pred, 0, None)
    
    mae = float(mean_absolute_error(y_true, y_pred_clean))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clean)))
    r2 = float(r2_score(y_true, y_pred_clean))
    
    non_zero = y_true > 0
    if non_zero.any():
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred_clean[non_zero]) / y_true[non_zero])) * 100)
    else:
        mape = 0.0
        
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
    }


def main():
    print("\n" + "=" * 70)
    print("  TORNEO DE MODELOS: INTELIGENCIA PREDICTIVA B2B (2021-2025)")
    print("=" * 70)

    config = {
        "project": {"name": "torneo_modelos", "seed": 42},
        "paths": {
            "data_dir": "./data",
            "artifacts_dir": "./artifacts",
            "outputs_dir": "./outputs",
        },
        "data": {
            "processed_path": "data/processed/dataset_enriquecido.csv",
            "train_years": [2021, 2022, 2023, 2024],
            "test_years": [2025],
            "target_column": "unidades_vendidas",
            "features": [
                "lag_1",
                "lag_2",
                "lag_12",
                "ma_3m",
                "price_gap_pct",
                "market_share_unidades",
                "temp_max_promedio",
                "precipitacion_mm",
                "ipc_indice",
                "tipo_cambio_pyg_usd",
            ],
        },
    }

    preprocessor = DataPreprocessor(config)
    X_train, y_train, X_test, y_test = preprocessor.prepare_splits()

    features_list = config["data"]["features"]

    competitors = [
        {
            "name": "Regresión Lineal (Baseline)",
            "family": "linear_model",
            "params": {},
        },
        {
            "name": "Árbol de Decisión (Decision Tree)",
            "family": "decision_tree",
            "params": {"max_depth": 10},
        },
        {
            "name": "Random Forest (100 árboles)",
            "family": "random_forest",
            "params": {"n_estimators": 100, "max_depth": 12, "n_jobs": -1},
        },
        {
            "name": "LightGBM (Boosting Ágil)",
            "family": "lightgbm",
            "params": {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.05},
        },
        {
            "name": "XGBoost (Extreme Boosting)",
            "family": "xgboost",
            "params": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05},
        },
    ]

    results = []
    feature_importances_dict = {}

    print("\n--- Ejecutando Torneo de Modelos ---")

    for comp in competitors:
        name = comp["name"]
        family = comp["family"]
        params = comp["params"]

        print(f"\n▶ Entrenando: {name}...")
        start_time = time.time()

        model_config = {
            "project": config["project"],
            "model": {"family": family, **params},
        }

        try:
            model = build_model(model_config)
            model.fit(X_train, y_train)
            train_duration = time.time() - start_time

            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)

            print(f"  ✓ Completado en {train_duration:.2f}s | RMSE: {metrics['rmse']:,} | MAE: {metrics['mae']:,} | MAPE: {metrics['mape']:.2f}% | R²: {metrics['r2']:.4f}")

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                total = np.sum(importances)
                normalized = (importances / total * 100) if total > 0 else importances
                feature_importances_dict[name] = {
                    feat: round(float(val), 2)
                    for feat, val in sorted(
                        zip(features_list, normalized), key=lambda x: x[1], reverse=True
                    )
                }
            elif hasattr(model, "coef_"):
                coefs = np.abs(model.coef_)
                total = np.sum(coefs)
                normalized = (coefs / total * 100) if total > 0 else coefs
                feature_importances_dict[name] = {
                    feat: round(float(val), 2)
                    for feat, val in sorted(
                        zip(features_list, normalized), key=lambda x: x[1], reverse=True
                    )
                }

            results.append({
                "Modelo": name,
                "Familia": family,
                "RMSE (unidades)": metrics["rmse"],
                "MAE (unidades)": metrics["mae"],
                "MAPE (%)": metrics["mape"],
                "R² Score": metrics["r2"],
                "Tiempo (s)": round(train_duration, 2),
            })

        except Exception as e:
            print(f"  ✗ Error al entrenar {name}: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE (unidades)", ascending=True).reset_index(drop=True)

    print("\n" + "=" * 85)
    print("  RESULTADOS FINALES DEL TORNEO DE MODELOS (ORDENADO POR PRECISION / RMSE)")
    print("=" * 85)
    print(results_df.to_string(index=False))
    print("=" * 85)

    reports_dir = os.path.join(config["paths"]["outputs_dir"], "reports")
    os.makedirs(reports_dir, exist_ok=True)

    json_report_path = os.path.join(reports_dir, "tournament_results.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": results,
            "feature_importances": feature_importances_dict,
            "train_period": "2021-2024",
            "test_period": "2025",
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Reporte JSON guardado en: {json_report_path}")

    if results:
        best_model_name = results_df.iloc[0]["Modelo"]
        print(f"\n[CHAMPION] MODELO GANADOR: {best_model_name}")
        if best_model_name in feature_importances_dict:
            print(f"\n[XAI] Explicabilidad Algoritmica (Feature Importance) para {best_model_name}:")
            for feat, pct in feature_importances_dict[best_model_name].items():
                bar = "#" * int(pct // 2)
                print(f"  * {feat:<24} : {pct:>5.2f}% {bar}")


if __name__ == "__main__":
    main()
