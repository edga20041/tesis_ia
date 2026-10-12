"""Reproduce el Torneo de Modelos original (ver scripts/experiment_tournament.py,
cuyo import de tesis_ia.data.preprocessor/tesis_ia.models.factory ya no existe tras
el refactor del paquete) con el MISMO dataset, features, partición temporal e
hiperparámetros documentados en ese script y en la Tabla 4.X de la tesis, y agrega:
1) un benchmark naive estacional (prediccion = lag_12, es decir "mismo mes, año
   anterior", sin ningun ajuste), y
2) WAPE y MASE (escala estacional m=12) junto al MAPE existente, para M3 de la
   devolucion de Opus.
"""
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

FEATURES = [
    "lag_1", "lag_2", "lag_12", "ma_3m", "price_gap_pct", "market_share_unidades",
    "temp_max_promedio", "precipitacion_mm", "ipc_indice", "tipo_cambio_pyg_usd",
]
TARGET = "unidades_vendidas"

df = pd.read_csv("data/processed/dataset_enriquecido.csv", sep=";", encoding="latin-1", decimal=",")
df[FEATURES] = df[FEATURES].fillna(0)
df[TARGET] = df[TARGET].fillna(0)

train_mask = df["anio"].isin([2021, 2022, 2023, 2024])
test_mask = df["anio"] == 2025

X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET]
X_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, TARGET]
print(f"Train: {len(X_train)} filas | Test (2025): {len(X_test)} filas")


def metrics(y_true, y_pred, mase_scale):
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nz = y_true > 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else 0.0
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)
    mase = float(mae / mase_scale) if mase_scale > 0 else float("nan")
    return dict(mae=mae, rmse=rmse, mape=mape, wape=wape, mase=mase, r2=r2)


# Escala MASE: error absoluto medio del naive estacional IN-SAMPLE (entrenamiento),
# tal como lo define Hyndman & Koehler (2006) para evitar que el denominador se mida
# sobre el mismo período que el numerador.
train_naive_scale = float(np.mean(np.abs(df.loc[train_mask, TARGET] - df.loc[train_mask, "lag_12"])))
print(f"Escala MASE (MAE naive estacional in-sample, train 2021-2024): {train_naive_scale:.2f}")

competitors = [
    ("Regresión Lineal (Baseline)", LinearRegression(), False),
    ("Árbol de Decisión", DecisionTreeRegressor(max_depth=10, random_state=42), True),
    ("Random Forest (100 árboles)", RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42), True),
    ("LightGBM", LGBMRegressor(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42, verbosity=-1), True),
    ("XGBoost", XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0), True),
]

rows = []
for name, model, _ in competitors:
    t0 = time.time()
    model.fit(X_train, y_train)
    dur = time.time() - t0
    y_pred = model.predict(X_test)
    m = metrics(y_test.to_numpy(), y_pred, train_naive_scale)
    m.update(modelo=name, tiempo=dur)
    rows.append(m)
    print(f"{name:35s} RMSE={m['rmse']:9.2f} MAE={m['mae']:8.2f} MAPE={m['mape']:8.2f}% WAPE={m['wape']:6.2f}% MASE={m['mase']:5.3f} R2={m['r2']:.4f} t={dur:.2f}s")

y_pred_naive = X_test["lag_12"].to_numpy()
m_naive = metrics(y_test.to_numpy(), y_pred_naive, train_naive_scale)
m_naive.update(modelo="Naïve Estacional (lag_12)", tiempo=0.0)
rows.append(m_naive)
print(f"{'Naïve Estacional (lag_12)':35s} RMSE={m_naive['rmse']:9.2f} MAE={m_naive['mae']:8.2f} MAPE={m_naive['mape']:8.2f}% WAPE={m_naive['wape']:6.2f}% MASE={m_naive['mase']:5.3f} R2={m_naive['r2']:.4f} t=0.00s")

out = pd.DataFrame(rows)[["modelo", "rmse", "mae", "mape", "wape", "mase", "r2", "tiempo"]]
out = out.sort_values("rmse")
out.to_csv("outputs/reports/tournament_m3_naive_wape.csv", index=False)
print("\nGuardado en outputs/reports/tournament_m3_naive_wape.csv")
print(out.to_string(index=False))
