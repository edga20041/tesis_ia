"""Evaluación de técnicas avanzadas: Target Logarítmico (log1p) y Días Hábiles."""

import calendar
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def get_working_days(year: int, month: int) -> int:
    """Calcula días laborales comerciales (Lunes a Sábado) del mes en Paraguay."""
    num_days = calendar.monthrange(int(year), int(month))[1]
    return sum(1 for day in range(1, num_days + 1) if datetime.date(int(year), int(month), day).weekday() < 6)

def main():
    features = ['lag_1', 'lag_2', 'lag_12', 'ma_3m', 'price_gap_pct', 'market_share_unidades', 'temp_max_promedio', 'precipitacion_mm', 'ipc_indice', 'tipo_cambio_pyg_usd']
    target = 'unidades_vendidas'
    path = "data/processed/dataset_enriquecido.csv"

    df = pd.read_csv(path, sep=";", decimal=",", encoding="latin1", low_memory=False)
    df_clean = df.dropna(subset=features + [target]).copy()
    
    df_clean['dias_laborables'] = df_clean.apply(lambda r: get_working_days(r['anio'], r['mes']), axis=1)

    train_mask = df_clean["anio"].isin([2021, 2022, 2023, 2024])
    test_mask = df_clean["anio"].isin([2025])

    X_train = df_clean.loc[train_mask, features]
    y_train = df_clean.loc[train_mask, target]
    X_test = df_clean.loc[test_mask, features]
    y_test = df_clean.loc[test_mask, target]

    features_ext = features + ['dias_laborables']
    X_train_ext = df_clean.loc[train_mask, features_ext]
    X_test_ext = df_clean.loc[test_mask, features_ext]

    print(f"[DATA] Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n1. Entrenando Random Forest Estándar (Lineal)...")
    rf_std = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1, random_state=42)
    rf_std.fit(X_train, y_train)
    y_pred_std = np.clip(rf_std.predict(X_test), 0, None)

    print("2. Entrenando Random Forest con Target Logarítmico log1p...")
    y_train_log = np.log1p(np.clip(y_train, 0, None))
    rf_log = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1, random_state=42)
    rf_log.fit(X_train, y_train_log)
    y_pred_log = np.expm1(np.clip(rf_log.predict(X_test), 0, None))

    print("3. Entrenando Random Forest con Target Logarítmico + Días Laborables...")
    rf_log_ext = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1, random_state=42)
    rf_log_ext.fit(X_train_ext, y_train_log)
    y_pred_log_ext = np.expm1(np.clip(rf_log_ext.predict(X_test_ext), 0, None))

    def evaluate(name, y_true, y_pred):
        non_zero = y_true > 0
        wmape_glob = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100
        mape_glob = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        
        q80 = np.quantile(y_true[non_zero], 0.80)
        mask_A = y_true >= q80
        mape_A = np.mean(np.abs((y_true[mask_A] - y_pred[mask_A]) / y_true[mask_A])) * 100
        wmape_A = (np.sum(np.abs(y_true[mask_A] - y_pred[mask_A])) / np.sum(y_true[mask_A])) * 100
        
        print(f"\n=== {name} ===")
        print(f"  WMAPE Global (Supply Chain): {wmape_glob:.2f}%")
        print(f"  MAPE No Ponderado: {mape_glob:.2f}%")
        print(f"  Clase A (Top 20% Volumen >= {q80:.0f} u): MAPE = {mape_A:.2f}% | WMAPE = {wmape_A:.2f}%")

    evaluate("1. RF Estándar", y_test, y_pred_std)
    evaluate("2. RF + Target Logarítmico (log1p)", y_test, y_pred_log)
    evaluate("3. RF + Target Logarítmico + Días Laborables", y_test, y_pred_log_ext)

if __name__ == '__main__':
    main()
