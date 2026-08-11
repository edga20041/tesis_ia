import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor

print("Iniciando conexión a DB y descarga de dataset...")
predictor = DbTrainerPredictor()
df = predictor.load_dataset_from_db()

print("Preparando el split de validación...")
df[predictor.FEATURE_COLS] = df[predictor.FEATURE_COLS].fillna(0)
df["periodo_date"] = pd.to_datetime(df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-01")
max_period_date = df["periodo_date"].max()

cutoff_date = max_period_date - pd.DateOffset(months=12)
train_mask = df["periodo_date"] < cutoff_date
test_mask = df["periodo_date"] >= cutoff_date

if not train_mask.any():
    cutoff_date = max_period_date - pd.DateOffset(months=6)
    train_mask = df["periodo_date"] < cutoff_date
    test_mask = df["periodo_date"] >= cutoff_date

X_train = df.loc[train_mask, predictor.FEATURE_COLS]
y_train = df.loc[train_mask, "unidades_vendidas"]
X_test = df.loc[test_mask, predictor.FEATURE_COLS]
y_test = df.loc[test_mask, "unidades_vendidas"]

print("Entrenando el modelo...")
model = LinearRegression()
model.fit(X_train, y_train)

print("Realizando predicciones...")
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, None) # Mismo clip a 0 que usás en producción

print("Exportando CSV...")
out_df = pd.DataFrame({"real": y_test, "prediccion": y_pred})
out_path = "predicciones_reales_vps.csv"
out_df.to_csv(out_path, index=False)

print(f"¡Éxito! Archivo guardado en: {out_path}")
