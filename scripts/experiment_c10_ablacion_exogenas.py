"""Ablacion con/sin variables exogenas sobre el modelo Champion (Random Forest),
para C10 de la devolucion de Opus: la importancia de variables ya documentada
(99,86% de peso en rezagos/media movil, seccion 7.5 del capitulo de Discusion)
sugiere que las exogenas aportan poco, pero no cuantifica el impacto real en
error si se las quita del todo. Usa el mismo dataset local y particion
(2021-2024 train / 2025 test) que el torneo de modelos (scripts/experiment_tournament_m3.py).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES_FULL = [
    "lag_1", "lag_2", "lag_12", "ma_3m", "price_gap_pct", "market_share_unidades",
    "temp_max_promedio", "precipitacion_mm", "ipc_indice", "tipo_cambio_pyg_usd",
]
FEATURES_SIN_EXOGENAS = [
    "lag_1", "lag_2", "lag_12", "ma_3m", "price_gap_pct", "market_share_unidades",
]
TARGET = "unidades_vendidas"

df = pd.read_csv("data/processed/dataset_enriquecido.csv", sep=";", encoding="latin-1", decimal=",")
df[FEATURES_FULL] = df[FEATURES_FULL].fillna(0)
df[TARGET] = df[TARGET].fillna(0)
df["categoria"] = df["categoria"].astype(str)

train_mask = df["anio"].isin([2021, 2022, 2023, 2024])
test_mask = df["anio"] == 2025


def metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nz = y_true > 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else 0.0
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)
    return dict(mae=mae, rmse=rmse, mape=mape, wape=wape, r2=r2)


def entrenar(features):
    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(df.loc[train_mask, features], df.loc[train_mask, TARGET])
    y_pred = model.predict(df.loc[test_mask, features])
    return model, y_pred


print("Entrenando Random Forest CON variables exogenas...")
modelo_full, pred_full = entrenar(FEATURES_FULL)
m_full = metrics(df.loc[test_mask, TARGET].to_numpy(), pred_full)
print(m_full)

print("\nEntrenando Random Forest SIN variables exogenas (solo lags/ma/competitivas)...")
modelo_sin, pred_sin = entrenar(FEATURES_SIN_EXOGENAS)
m_sin = metrics(df.loc[test_mask, TARGET].to_numpy(), pred_sin)
print(m_sin)

print("\n--- Comparacion global ---")
for k in ["rmse", "mae", "mape", "wape", "r2"]:
    full_v, sin_v = m_full[k], m_sin[k]
    delta = (full_v - sin_v) / sin_v * 100 if sin_v != 0 else float("nan")
    print(f"{k:6s}: con_exogenas={full_v:10.3f}  sin_exogenas={sin_v:10.3f}  variacion={delta:+.2f}%")

test_df = df.loc[test_mask].copy()
test_df["pred_full"] = pred_full
test_df["pred_sin"] = pred_sin

resumen_cat = []
for cat, g in test_df.groupby("categoria"):
    if g[TARGET].sum() < 1000:
        continue
    wape_full = float(np.sum(np.abs(g[TARGET] - np.clip(g["pred_full"], 0, None))) / np.sum(np.abs(g[TARGET])) * 100)
    wape_sin = float(np.sum(np.abs(g[TARGET] - np.clip(g["pred_sin"], 0, None))) / np.sum(np.abs(g[TARGET])) * 100)
    resumen_cat.append((cat, g[TARGET].sum(), wape_full, wape_sin, wape_full - wape_sin))

resumen_cat.sort(key=lambda r: -r[1])
print("\n--- WAPE por categoria (top 10 por volumen) ---")
print(f"{'categoria':30s} {'unidades':>12s} {'wape_con':>9s} {'wape_sin':>9s} {'delta_pp':>9s}")
for cat, vol, wf, ws, d in resumen_cat[:10]:
    print(f"{cat:30s} {vol:12.0f} {wf:9.2f} {ws:9.2f} {d:+9.2f}")

pd.DataFrame(resumen_cat, columns=["categoria", "unidades", "wape_con_exogenas", "wape_sin_exogenas", "delta_pp"]).to_csv(
    "outputs/reports/c10_ablacion_exogenas_por_categoria.csv", index=False)
print("\nGuardado en outputs/reports/c10_ablacion_exogenas_por_categoria.csv")
