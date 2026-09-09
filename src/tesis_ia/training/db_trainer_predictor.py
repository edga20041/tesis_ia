"""Módulo de entrenamiento del Modelo Champion (Random Forest / Regresión Lineal) y generación de predicciones a 12 meses directamente en MySQL."""

from __future__ import annotations

import gc
import json
import uuid
import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text

from tesis_ia.utils.db_connector import get_engine

# Variable global en memoria para caché del modelo entrenado
_CACHED_MODEL: Optional[Any] = None

# Caché del especialista de target relativo del modelo híbrido. Se mantiene separado
# de _CACHED_MODEL a propósito: el endpoint XAI y el simulador siguen leyendo el
# modelo logarítmico, cuyo espacio de features no cambió.
_CACHED_RATIO_MODEL: Optional[Any] = None


class DbTrainerPredictor:

    FEATURE_COLS = [
        "lag_1", "lag_2", "lag_12", "ma_3m",
        "temp_max_promedio", "temp_media_promedio", "precipitacion_mm",
        "ipc_indice", "ipc_var_interanual", "tipo_cambio_pyg_usd",
        "cantidad_feriados", "tiene_festivo_comercial"
    ]

    # Features adicionales del especialista de target relativo. Las variables
    # contemporáneas van desplazadas un mes porque market_share_unidades y
    # price_gap_pct del mes t se calculan con las ventas de ese mismo mes: usarlas
    # sin desplazar sería fuga de datos (el modelo vería la respuesta).
    EXTRA_FEATURE_COLS = [
        "mes_sin", "mes_cos",
        "precio_unitario_lag1", "price_gap_lag1", "market_share_lag1",
        "log_base",
    ]

    # Techo relativo del modelo híbrido: la predicción del especialista de ratio no
    # puede exceder K veces la del modelo logarítmico. Acota el modo de falla del
    # target relativo en SKUs con base histórica obsoleta (media móvil alta y demanda
    # derrumbada), donde llegaba a un APE medio de 804%.
    # K=1.25 fue elegido por validación anidada sobre la ventana 2023-12/2024-12 y
    # evaluado sin reajuste sobre 2024-12/2025-12.
    K_TECHO_RELATIVO = 1.25

    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def load_dataset_from_db(self) -> pd.DataFrame:
        query = """
            SELECT 
                vm.id AS id_venta,
                vm.id_producto,
                vm.id_empresa,
                vm.anio,
                vm.mes,
                vm.unidades_vendidas,
                vm.monto_facturado,
                vm.precio_unitario,
                COALESCE(vm.lag_1, 0) AS lag_1,
                COALESCE(vm.lag_2, 0) AS lag_2,
                COALESCE(vm.lag_12, 0) AS lag_12,
                COALESCE(vm.ma_3m, 0) AS ma_3m,
                COALESCE(vm.market_share_unidades, 0) AS market_share_unidades,
                COALESCE(vm.price_gap_pct, 0) AS price_gap_pct,
                COALESCE(vc.temp_max_promedio, 30.0) AS temp_max_promedio,
                COALESCE(vc.temp_media_promedio, 25.0) AS temp_media_promedio,
                COALESCE(vc.precipitacion_mm, 100.0) AS precipitacion_mm,
                COALESCE(vmac.ipc_indice, 100.0) AS ipc_indice,
                COALESCE(vmac.ipc_var_interanual, 4.5) AS ipc_var_interanual,
                COALESCE(vmac.tipo_cambio_pyg_usd, 7200.0) AS tipo_cambio_pyg_usd,
                COALESCE(vcal.cantidad_feriados, 0) AS cantidad_feriados,
                COALESCE(vcal.tiene_festivo_comercial, 0) AS tiene_festivo_comercial
            FROM ventas_mensuales vm
            LEFT JOIN variables_climaticas vc ON vm.anio = vc.anio AND vm.mes = vc.mes
            LEFT JOIN variables_macroeconomicas vmac ON vm.anio = vmac.anio AND vm.mes = vmac.mes
            LEFT JOIN variables_calendario vcal ON vm.anio = vcal.anio AND vm.mes = vcal.mes
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df

    @staticmethod
    def calcular_base_escala(ma_3m, lag_1, lag_12):
        """Nivel histórico de referencia del SKU, denominador del target relativo.

        Prioriza la media móvil y cae a lag_1 y lag_12 cuando la serie es corta.
        Debe calcularse igual en entrenamiento y en inferencia o el target relativo
        se reconstruye contra una escala distinta a la aprendida.
        """
        base = np.asarray(ma_3m, dtype=float).copy()
        for alt in (lag_1, lag_12):
            faltante = base <= 0
            if not faltante.any():
                break
            base[faltante] = np.asarray(alt, dtype=float)[faltante]
        return np.clip(base, 0, None)

    def derive_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["id_producto", "id_empresa", "anio", "mes"]).reset_index(drop=True)

        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12.0)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12.0)

        grupo = df.groupby(["id_producto", "id_empresa"], sort=False)
        df["precio_unitario_lag1"] = grupo["precio_unitario"].shift(1).fillna(0.0)
        df["price_gap_lag1"] = grupo["price_gap_pct"].shift(1).fillna(0.0)
        df["market_share_lag1"] = grupo["market_share_unidades"].shift(1).fillna(0.0)

        df["base_escala"] = self.calcular_base_escala(df["ma_3m"], df["lag_1"], df["lag_12"])
        df["log_base"] = np.log1p(df["base_escala"])
        return df

    @classmethod
    def aplicar_techo_relativo(cls, pred_log: np.ndarray, pred_ratio: np.ndarray) -> np.ndarray:
        return np.minimum(pred_ratio, cls.K_TECHO_RELATIVO * np.maximum(pred_log, 1.0))

    def train_and_evaluate(self, model_family: str = "random_forest") -> Tuple[Any, Dict[str, float], str, str]:
        global _CACHED_MODEL, _CACHED_RATIO_MODEL
        df = self.load_dataset_from_db()
        if df.empty:
            raise ValueError("No hay datos en ventas_mensuales para entrenar.")

        df[self.FEATURE_COLS] = df[self.FEATURE_COLS].fillna(0)
        df = self.derive_extra_features(df)

        df["periodo_date"] = pd.to_datetime(df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-01")
        max_period_date = df["periodo_date"].max()

        cutoff_date = max_period_date - pd.DateOffset(months=12)

        train_mask = df["periodo_date"] < cutoff_date
        test_mask = df["periodo_date"] >= cutoff_date

        if not train_mask.any():
            cutoff_date = max_period_date - pd.DateOffset(months=6)
            train_mask = df["periodo_date"] < cutoff_date
            test_mask = df["periodo_date"] >= cutoff_date

        X_train = df.loc[train_mask, self.FEATURE_COLS]
        y_train = df.loc[train_mask, "unidades_vendidas"]
        X_test = df.loc[test_mask, self.FEATURE_COLS]
        y_test = df.loc[test_mask, "unidades_vendidas"]

        if model_family == "random_forest":
            # Hiperparametros ajustados por barrido real contra la BD de produccion (backtest de
            # ultimos 12 meses observados): n_estimators=50/max_depth=10 daba MAPE=33.31%/WMAPE=12.79%
            # con ~286MB de RAM por proceso. Se probo primero max_depth=None (sin limite), que bajaba
            # a MAPE=15.38% pero tumbaba el contenedor por OOM en el VPS real (~5.8GB de RSS, confirmado
            # con `dmesg | grep "Out of memory"` tras 3 caidas reales en produccion) - max_depth no acota
            # el tamano del arbol de forma predecible con ~500k filas de entrenamiento.
            # Fix: max_leaf_nodes acota el arbol de forma DIRECTA (independiente de como se ramifiquen
            # los datos). n_estimators=100/max_leaf_nodes=10000 da MAPE=15.38%->16.92%/WMAPE=6.77%->9.37%
            # (casi toda la mejora de la config sin limite) con solo ~627MB de RSS medido (9x menos que
            # la version que rompio produccion), dejando margen real en un VPS de 7.8GB compartido con
            # MySQL y tesis_back.
            print("[TRAIN] Entrenando Champion: Random Forest Regressor (n_estimators=100, max_leaf_nodes=10000, n_jobs=-1)...")
            model = RandomForestRegressor(
                n_estimators=100,
                max_leaf_nodes=10000,
                n_jobs=-1,
                random_state=42
            )
        else:
            print("[TRAIN] Entrenando Baseline: Regresion Lineal Multivariada...")
            model = LinearRegression()

        y_train_log = np.log1p(np.clip(y_train, 0, None))
        model.fit(X_train, y_train_log)

        # Guardar en memoria RAM para uso inmediato en simulaciones
        _CACHED_MODEL = model

        y_pred_log = model.predict(X_test)
        y_pred_solo_log = np.clip(np.expm1(np.clip(y_pred_log, 0, None)), 0, None)

        # Especialista de target relativo: aprende log1p(y) - log1p(base) en vez de
        # log1p(y). Al predecir un ratio contra la escala del SKU en lugar de un nivel
        # absoluto, rompe el techo de extrapolación del Random Forest, que nunca puede
        # devolver un valor por encima del máximo visto en entrenamiento y por eso
        # sub-predecía sistemáticamente los SKUs de alto volumen en crecimiento.
        y_pred = y_pred_solo_log
        if model_family == "random_forest":
            cols_ratio = self.FEATURE_COLS + self.EXTRA_FEATURE_COLS
            log_base_train = np.log1p(df.loc[train_mask, "base_escala"].to_numpy(dtype=float))
            log_base_test = np.log1p(df.loc[test_mask, "base_escala"].to_numpy(dtype=float))

            print("[TRAIN] Entrenando especialista de target relativo (max_leaf_nodes=30000)...")
            ratio_model = RandomForestRegressor(
                n_estimators=100,
                max_leaf_nodes=30000,
                n_jobs=-1,
                random_state=42
            )
            ratio_model.fit(df.loc[train_mask, cols_ratio], y_train_log - log_base_train)
            _CACHED_RATIO_MODEL = ratio_model

            pred_ratio_log = ratio_model.predict(df.loc[test_mask, cols_ratio]) + log_base_test
            y_pred_ratio = np.clip(np.expm1(np.clip(pred_ratio_log, 0, None)), 0, None)
            y_pred = self.aplicar_techo_relativo(y_pred_solo_log, y_pred_ratio)

        mae = float(mean_absolute_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred))) if len(y_test) > 0 else 0.0

        # MAPE evitando división por cero
        if len(y_test) > 0 and (y_test > 0).any():
            non_zero = y_test > 0
            mape = float(np.mean(np.abs((y_test[non_zero] - y_pred[non_zero]) / y_test[non_zero])) * 100)
        else:
            mape = 0.0

        fecha_desde = f"{df['anio'].min()}-01-01"
        fecha_hasta = f"{df['anio'].max()}-12-31"

        metrics = {"mae": mae, "rmse": rmse, "mape": mape}
        print(f"[TRAIN] Modelo {model_family} entrenado exitosamente (Ventana: Train < {cutoff_date.strftime('%Y-%m')}, Val >= {cutoff_date.strftime('%Y-%m')}). MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return model, metrics, fecha_desde, fecha_hasta

    def save_model_version(self, version: str, metrics: Dict[str, float],
                           desde: str, hasta: str, id_empresa: Optional[str] = None,
                           model_family: str = "random_forest") -> Dict[str, str]:
        sql_get_empresas = text("SELECT id FROM empresa WHERE activo = 1")
        sql_deactivate = text("UPDATE modelos_versiones SET activo = 0 WHERE id_empresa = :id_empresa")

        sql_insert = text("""
            INSERT INTO modelos_versiones (
                id, id_empresa, version, fecha_entrenamiento, periodo_datos_desde,
                periodo_datos_hasta, mae, rmse, mape, hiperparametros, activo
            ) VALUES (
                :id, :id_empresa, :version, NOW(), :desde,
                :hasta, :mae, :rmse, :mape, :hiperparametros, 1
            )
            ON DUPLICATE KEY UPDATE
                fecha_entrenamiento = NOW(),
                mae = VALUES(mae),
                rmse = VALUES(rmse),
                mape = VALUES(mape),
                hiperparametros = VALUES(hiperparametros),
                activo = 1
        """)

        company_model_map: Dict[str, str] = {}

        if model_family == "random_forest":
            hiperparams = json.dumps({"family": "RandomForestRegressor", "n_estimators": 100, "max_leaf_nodes": 10000, "algorithm": "scikit-learn"})
        else:
            hiperparams = json.dumps({"family": "LinearRegression", "algorithm": "scikit-learn"})

        with self.engine.begin() as conn:
            if id_empresa:
                empresas = [id_empresa]
            else:
                empresas = [row[0] for row in conn.execute(sql_get_empresas).fetchall()]
                if not empresas:
                    empresas = [str(uuid.uuid4())]

            for emp_id in empresas:
                model_id = str(uuid.uuid4())
                conn.execute(sql_deactivate, {"id_empresa": emp_id})
                conn.execute(sql_insert, {
                    "id": model_id,
                    "id_empresa": emp_id,
                    "version": version,
                    "desde": desde,
                    "hasta": hasta,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "hiperparametros": hiperparams
                })

            rows = conn.execute(text("SELECT id_empresa, id FROM modelos_versiones WHERE activo = 1")).fetchall()
            for r_emp, r_id in rows:
                company_model_map[r_emp] = r_id

        print(f"[OK] Modelo '{version}' guardado como ACTIVO para {len(company_model_map)} empresa(s).")
        return company_model_map

    def generate_and_save_predictions(self, model: Any,
                                       company_model_map: Union[Dict[str, str], str]) -> int:
        global _CACHED_RATIO_MODEL
        query_prods = """
            SELECT DISTINCT 
                p.id AS id_producto,
                m.id_empresa,
                p.valor_unitario
            FROM productos p
            JOIN marcas m ON p.id_marca = m.id
        """
        query_stats = """
            SELECT
                r.id_producto,
                s.avg_unidades,
                COALESCE(r.lag_1, r.unidades_vendidas) AS last_lag_1,
                COALESCE(r.lag_2, r.unidades_vendidas) AS last_lag_2,
                COALESCE(r.ma_3m, r.unidades_vendidas) AS last_ma_3m,
                COALESCE(r.market_share_unidades, 0) AS last_market_share,
                COALESCE(r.price_gap_pct, 0) AS last_price_gap,
                r.anio AS last_anio,
                r.mes AS last_mes
            FROM (
                SELECT
                    id_producto, anio, mes, unidades_vendidas, lag_1, lag_2, ma_3m,
                    market_share_unidades, price_gap_pct,
                    ROW_NUMBER() OVER (PARTITION BY id_producto ORDER BY anio DESC, mes DESC) AS rn
                FROM ventas_mensuales
            ) r
            JOIN (
                SELECT id_producto, AVG(unidades_vendidas) AS avg_unidades
                FROM ventas_mensuales
                GROUP BY id_producto
            ) s ON s.id_producto = r.id_producto
            WHERE r.rn = 1
        """
        query_monthly_seasonality = """
            SELECT 
                id_producto,
                mes,
                AVG(unidades_vendidas) AS avg_mes
            FROM ventas_mensuales
            GROUP BY id_producto, mes
        """

        query_exo_exact = """
            SELECT 
                vc.anio, vc.mes,
                COALESCE(vc.temp_max_promedio, 30.0) AS temp_max_promedio,
                COALESCE(vc.temp_media_promedio, 25.0) AS temp_media_promedio,
                COALESCE(vc.precipitacion_mm, 100.0) AS precipitacion_mm,
                COALESCE(vmac.ipc_indice, 100.0) AS ipc_indice,
                COALESCE(vmac.ipc_var_interanual, 4.5) AS ipc_var_interanual,
                COALESCE(vmac.tipo_cambio_pyg_usd, 7200.0) AS tipo_cambio_pyg_usd,
                COALESCE(vcal.cantidad_feriados, 0) AS cantidad_feriados,
                COALESCE(vcal.tiene_festivo_comercial, 0) AS tiene_festivo_comercial
            FROM variables_climaticas vc
            LEFT JOIN variables_macroeconomicas vmac ON vc.anio = vmac.anio AND vc.mes = vmac.mes
            LEFT JOIN variables_calendario vcal ON vc.anio = vcal.anio AND vc.mes = vcal.mes
        """

        query_exo_by_month = """
            SELECT 
                vc.mes,
                AVG(COALESCE(vc.temp_max_promedio, 30.0)) AS temp_max_promedio,
                AVG(COALESCE(vc.temp_media_promedio, 25.0)) AS temp_media_promedio,
                AVG(COALESCE(vc.precipitacion_mm, 100.0)) AS precipitacion_mm,
                MAX(COALESCE(vmac.ipc_indice, 100.0)) AS ipc_indice,
                AVG(COALESCE(vmac.ipc_var_interanual, 4.5)) AS ipc_var_interanual,
                MAX(COALESCE(vmac.tipo_cambio_pyg_usd, 7200.0)) AS tipo_cambio_pyg_usd,
                ROUND(AVG(COALESCE(vcal.cantidad_feriados, 0))) AS cantidad_feriados,
                MAX(COALESCE(vcal.tiene_festivo_comercial, 0)) AS tiene_festivo_comercial
            FROM variables_climaticas vc
            LEFT JOIN variables_macroeconomicas vmac ON vc.anio = vmac.anio AND vc.mes = vmac.mes
            LEFT JOIN variables_calendario vcal ON vc.anio = vcal.anio AND vc.mes = vcal.mes
            GROUP BY vc.mes
        """

        with self.engine.connect() as conn:
            prods_df = pd.read_sql(query_prods, conn)
            stats_df = pd.read_sql(query_stats, conn)
            seasonality_df = pd.read_sql(query_monthly_seasonality, conn)
            exo_exact_df = pd.read_sql(query_exo_exact, conn).set_index(["anio", "mes"])
            exo_month_df = pd.read_sql(query_exo_by_month, conn).set_index("mes")

        if prods_df.empty:
            print("[WARN] No hay productos para generar predicciones.")
            return 0

        prods_df["id_producto_clean"] = prods_df["id_producto"].astype(str).str.strip().str.lower()
        stats_df["id_producto_clean"] = stats_df["id_producto"].astype(str).str.strip().str.lower()
        seasonality_df["id_producto_clean"] = seasonality_df["id_producto"].astype(str).str.strip().str.lower()

        prods_merged = prods_df.merge(stats_df, on="id_producto_clean", how="left", suffixes=("", "_stat"))
        seasonality_map = seasonality_df.set_index(["id_producto_clean", "mes"])["avg_mes"].to_dict()

        # Inferencia vectorizada de alta velocidad para todo el catálogo
        n_prods = len(prods_merged)
        print(f"[PREDICT] Iniciando inferencia vectorizada de alta velocidad para {n_prods:,} productos a 12 meses...")

        ids_prod = prods_merged["id_producto"].to_numpy()
        ids_prod_clean = prods_merged["id_producto_clean"].to_numpy()
        ids_emp = prods_merged["id_empresa"].to_numpy()
        val_unitarios = prods_merged["valor_unitario"].fillna(10000.0).to_numpy(dtype=np.float32)

        # Última observación conocida de las variables contemporáneas. En entrenamiento
        # el especialista de ratio las ve desplazadas un mes, así que el valor más
        # reciente del histórico es justamente lo que corresponde usar al proyectar.
        last_market_share = prods_merged.get(
            "last_market_share", pd.Series(0.0, index=prods_merged.index)
        ).fillna(0.0).to_numpy(dtype=np.float32)
        last_price_gap = prods_merged.get(
            "last_price_gap", pd.Series(0.0, index=prods_merged.index)
        ).fillna(0.0).to_numpy(dtype=np.float32)

        base_units_series = prods_merged["avg_unidades"].fillna(50.0)
        l1_series = prods_merged["last_lag_1"].fillna(base_units_series)
        l2_series = prods_merged["last_lag_2"].fillna(base_units_series)
        ma3_series = prods_merged["last_ma_3m"].fillna(base_units_series)

        base_units = base_units_series.to_numpy(dtype=np.float32)
        l1 = l1_series.to_numpy(dtype=np.float32)
        l2 = l2_series.to_numpy(dtype=np.float32)
        ma3 = ma3_series.to_numpy(dtype=np.float32)

        start_years = prods_merged["last_anio"].fillna(2025).astype(int).to_numpy()
        start_months = prods_merged["last_mes"].fillna(12).astype(int).to_numpy()

        default_model_id = company_model_map if isinstance(company_model_map, str) else next(iter(company_model_map.values()), str(uuid.uuid4()))
        if isinstance(company_model_map, dict):
            model_ids = np.array([company_model_map.get(emp, default_model_id) for emp in ids_emp])
        else:
            model_ids = np.array([default_model_id] * n_prods)

        curr_y = start_years.copy()
        curr_m = start_months.copy()

        proyecciones = []

        for step in range(1, 13):
            curr_m = curr_m + 1
            overflow = curr_m > 12
            curr_y[overflow] += 1
            curr_m[overflow] = 1

            l12 = np.empty(n_prods, dtype=np.float32)
            t_max = np.empty(n_prods, dtype=np.float32)
            t_med = np.empty(n_prods, dtype=np.float32)
            precip = np.empty(n_prods, dtype=np.float32)
            ipc_ind = np.empty(n_prods, dtype=np.float32)
            ipc_var = np.empty(n_prods, dtype=np.float32)
            tc_usd = np.empty(n_prods, dtype=np.float32)
            n_feriados = np.empty(n_prods, dtype=np.float32)
            has_festivo = np.empty(n_prods, dtype=np.float32)

            for j in range(n_prods):
                p_c = ids_prod_clean[j]
                m_val = curr_m[j]
                y_val = curr_y[j]

                l12[j] = seasonality_map.get((p_c, m_val), base_units[j])

                if (y_val, m_val) in exo_exact_df.index:
                    e = exo_exact_df.loc[(y_val, m_val)]
                    if isinstance(e, pd.DataFrame): e = e.iloc[0]
                    t_max[j] = float(e["temp_max_promedio"])
                    t_med[j] = float(e["temp_media_promedio"])
                    precip[j] = float(e["precipitacion_mm"])
                    ipc_ind[j] = float(e["ipc_indice"])
                    ipc_var[j] = float(e["ipc_var_interanual"])
                    tc_usd[j] = float(e["tipo_cambio_pyg_usd"])
                    n_feriados[j] = float(e["cantidad_feriados"])
                    has_festivo[j] = float(e["tiene_festivo_comercial"])
                elif m_val in exo_month_df.index:
                    e = exo_month_df.loc[m_val]
                    if isinstance(e, pd.DataFrame): e = e.iloc[0]
                    t_max[j] = float(e["temp_max_promedio"])
                    t_med[j] = float(e["temp_media_promedio"])
                    precip[j] = float(e["precipitacion_mm"])
                    ipc_ind[j] = float(e["ipc_indice"])
                    ipc_var[j] = float(e["ipc_var_interanual"])
                    tc_usd[j] = float(e["tipo_cambio_pyg_usd"])
                    n_feriados[j] = float(e["cantidad_feriados"])
                    has_festivo[j] = float(e["tiene_festivo_comercial"])
                else:
                    t_max[j], t_med[j], precip[j] = 30.0, 25.0, 100.0
                    ipc_ind[j], ipc_var[j], tc_usd[j] = 100.0, 4.5, 7500.0
                    n_feriados[j], has_festivo[j] = 0.0, 0.0

            # Matriz de características para todos los productos en este mes: shape (N, 12)
            X_step = np.column_stack([
                l1, l2, l12, ma3,
                t_max, t_med, precip,
                ipc_ind, ipc_var, tc_usd,
                n_feriados, has_festivo
            ])

            preds_raw_log = model.predict(X_step)
            preds_raw = np.expm1(np.clip(preds_raw_log, 0, None))

            # Modulación Estacional: preserva las oscilaciones orgánicas del mercado evitando que el árbol se estanque
            seasonal_ratio = np.clip(l12 / np.maximum(base_units, 1.0), 0.70, 1.40)

            if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                ratio_model = _CACHED_RATIO_MODEL
                if ratio_model is not None:
                    # Matriz extendida del especialista de ratio. El orden de las columnas
                    # debe coincidir con FEATURE_COLS + EXTRA_FEATURE_COLS.
                    base_escala = self.calcular_base_escala(ma3, l1, l12)
                    log_base = np.log1p(base_escala)
                    X_step_ratio = np.column_stack([
                        X_step,
                        np.sin(2 * np.pi * curr_m / 12.0),
                        np.cos(2 * np.pi * curr_m / 12.0),
                        val_unitarios, last_price_gap, last_market_share,
                        log_base,
                    ])
                    tree_source, X_trees, desplazamiento = ratio_model, X_step_ratio, log_base
                else:
                    # Sin especialista en caché (por ejemplo, tras reiniciar el proceso sin
                    # reentrenar) se degrada al comportamiento logarítmico previo.
                    tree_source, X_trees, desplazamiento = model, X_step, 0.0

                tree_preds_raw = np.array([tree.predict(X_trees) for tree in tree_source.estimators_])  # shape: (n_arboles, N)
                tree_preds_units = np.expm1(np.clip(tree_preds_raw + desplazamiento, 0, None))  # shape: (n_arboles, N)
                tree_preds_mod = tree_preds_units * seasonal_ratio  # shape: (n_arboles, N)

                if ratio_model is not None:
                    # Techo relativo aplicado a la distribución completa, no solo al valor
                    # central: así los percentiles y la banda de confianza quedan coherentes
                    # con la predicción publicada.
                    techo = self.K_TECHO_RELATIVO * np.maximum(preds_raw * seasonal_ratio, 1.0)
                    tree_preds_mod = np.minimum(tree_preds_mod, techo)

                pred_p10 = np.percentile(tree_preds_mod, 10, axis=0)
                pred_p50 = np.percentile(tree_preds_mod, 50, axis=0)
                pred_p90 = np.percentile(tree_preds_mod, 90, axis=0)

                pred_unidades = np.where(
                    pred_p50 > 0,
                    np.round(pred_p50, 2),
                    np.maximum(1.0, np.round(l12 * (1.0 + has_festivo * 0.15 - precip * 0.0005), 2))
                )
                pred_inf = np.maximum(0.0, np.round(pred_p10, 2))
                pred_sup = np.round(np.maximum(pred_unidades, pred_p90), 2)
            else:
                preds_raw_log = model.predict(X_step)
                preds_raw = np.expm1(np.clip(preds_raw_log, 0, None))
                pred_unidades = np.where(
                    preds_raw > 0,
                    np.round(preds_raw * seasonal_ratio, 2),
                    np.maximum(1.0, np.round(l12 * (1.0 + has_festivo * 0.15 - precip * 0.0005), 2))
                )
                pred_inf = np.round(pred_unidades * 0.85, 2)
                pred_sup = np.round(pred_unidades * 1.15, 2)

            for j in range(n_prods):
                p_unidades = float(pred_unidades[j])
                val_u = float(val_unitarios[j])
                p_y = int(curr_y[j])
                p_m = int(curr_m[j])
                p_id = ids_prod[j]
                proyecciones.append({
                    "id": f"{p_id}_{p_y}_{p_m}",
                    "id_empresa": ids_emp[j],
                    "id_producto": p_id,
                    "id_modelo": model_ids[j],
                    "anio": p_y,
                    "mes": p_m,
                    "unidades_predichas": p_unidades,
                    "monto_predicho": round(p_unidades * val_u, 2),
                    "intervalo_confianza_inf": float(pred_inf[j]),
                    "intervalo_confianza_sup": float(pred_sup[j]),
                })

            l2 = l1.copy()
            l1 = pred_unidades.copy()
            # Misma fórmula que en entrenamiento (feature_calculator.py): ma_3m = (lag_1 + lag_2) / 2.
            # Antes era (l1+l2+ma3_anterior)/3, una fórmula distinta a la aprendida por el modelo —
            # ma_3m es el feature dominante, así que la discrepancia distorsionaba fuerte los primeros
            # meses de cada pronóstico tras cualquier cambio de nivel de demanda.
            ma3 = np.round((l1 + l2) / 2.0, 2)

        sql_delete = text("DELETE FROM predicciones")
        sql_pred = text("""
            INSERT INTO predicciones (
                id, id_empresa, id_producto, id_modelo, anio, mes,
                unidades_predichas, monto_predicho, intervalo_confianza_inf,
                intervalo_confianza_sup, fecha_generacion
            ) VALUES (
                :id, :id_empresa, :id_producto, :id_modelo, :anio, :mes,
                :unidades_predichas, :monto_predicho, :intervalo_confianza_inf,
                :intervalo_confianza_sup, NOW()
            )
        """)

        try:
            with self.engine.begin() as conn:
                conn.execute(sql_delete)
                batch_size = 10000
                for i in range(0, len(proyecciones), batch_size):
                    batch = proyecciones[i:i + batch_size]
                    conn.execute(sql_pred, batch)
            print(f"[OK] {len(proyecciones)} predicciones dinámicas con estacionalidad insertadas en la tabla predicciones.")
            return len(proyecciones)
        except Exception as e:
            print(f"[ERROR] Falló la inserción de predicciones en MySQL: {e}")
            raise e
        finally:
            # El especialista de ratio solo interviene aquí: ni el simulador ni el
            # endpoint XAI lo consultan. Liberarlo apenas termina el cierre devuelve el
            # consumo en régimen al nivel previo al híbrido, dejando el pico de dos
            # ensambles acotado a la ventana del cierre mensual. El próximo cierre lo
            # reentrena de todas formas antes de volver a predecir.
            _CACHED_RATIO_MODEL = None
            gc.collect()
            print("[MEM] Especialista de ratio liberado tras el cierre.")

    def simulate_product_scenarios(self, id_producto: str, precio_pct: float = 0.0,
                                   usd_pct: float = 0.0, ipc_pct: float = 0.0) -> List[Dict[str, Any]]:
        global _CACHED_MODEL
        if _CACHED_MODEL is None:
            print("[INFO] Reentrenando modelo para simulación (primera ejecución RAM)...")
            model, metrics, _, _ = self.train_and_evaluate()
        else:
            model = _CACHED_MODEL

        id_clean = id_producto.strip().lower()

        # El frontend identifica los productos por SKU (EAN), mientras que las
        # tablas de ventas se indexan por el UUID interno (productos.id). Aceptamos
        # cualquiera de los dos: matcheamos por id o por sku, y mas abajo resolvemos
        # el UUID real para el resto de los queries.
        query_prod = text("""
            SELECT
                p.id AS id_producto,
                p.valor_unitario,
                m.id_empresa
            FROM productos p
            JOIN marcas m ON p.id_marca = m.id
            WHERE LOWER(p.id) = :id_clean OR LOWER(p.sku) = :id_clean
            LIMIT 1
        """)

        # last_lag_1/last_lag_2/last_ma_3m: valores de la fila REAL más reciente del producto (no un
        # AVG() de toda su historia, que arranca la simulación en el nivel equivocado para productos
        # con tendencia — mismo fix que query_stats de generate_and_save_predictions).
        query_stat = text("""
            SELECT
                (SELECT AVG(unidades_vendidas) FROM ventas_mensuales WHERE LOWER(id_producto) = :id_clean) AS avg_unidades,
                COALESCE(lag_1, unidades_vendidas) AS last_lag_1,
                COALESCE(lag_2, unidades_vendidas) AS last_lag_2,
                COALESCE(ma_3m, unidades_vendidas) AS last_ma_3m,
                anio AS last_anio,
                mes AS last_mes
            FROM ventas_mensuales
            WHERE LOWER(id_producto) = :id_clean
            ORDER BY anio DESC, mes DESC
            LIMIT 1
        """)

        query_seasonality = text("""
            SELECT mes, AVG(unidades_vendidas) AS avg_mes
            FROM ventas_mensuales
            WHERE LOWER(id_producto) = :id_clean
            GROUP BY mes
        """)

        query_exo = text("""
            SELECT 
                vc.mes,
                AVG(COALESCE(vc.temp_max_promedio, 30.0)) AS temp_max_promedio,
                AVG(COALESCE(vc.temp_media_promedio, 25.0)) AS temp_media_promedio,
                AVG(COALESCE(vc.precipitacion_mm, 100.0)) AS precipitacion_mm,
                MAX(COALESCE(vmac.ipc_indice, 100.0)) AS ipc_indice,
                AVG(COALESCE(vmac.ipc_var_interanual, 4.5)) AS ipc_var_interanual,
                MAX(COALESCE(vmac.tipo_cambio_pyg_usd, 7200.0)) AS tipo_cambio_pyg_usd,
                ROUND(AVG(COALESCE(vcal.cantidad_feriados, 0))) AS cantidad_feriados,
                MAX(COALESCE(vcal.tiene_festivo_comercial, 0)) AS tiene_festivo_comercial
            FROM variables_climaticas vc
            LEFT JOIN variables_macroeconomicas vmac ON vc.anio = vmac.anio AND vc.mes = vmac.mes
            LEFT JOIN variables_calendario vcal ON vc.anio = vcal.anio AND vc.mes = vcal.mes
            GROUP BY vc.mes
        """)

        with self.engine.connect() as conn:
            prod_row = conn.execute(query_prod, {"id_clean": id_clean}).fetchone()
            if not prod_row:
                raise ValueError(f"No se encontró el producto con ID '{id_producto}' en la base de datos.")

            # prod_row[0] es el UUID interno real (p.id), venga el request por id o por
            # sku. Los queries sobre ventas_mensuales SIEMPRE se filtran por ese UUID,
            # nunca por el sku (que no existe en esa tabla).
            id_real = str(prod_row[0]).strip().lower()

            stat_row = conn.execute(query_stat, {"id_clean": id_real}).fetchone()
            seas_rows = conn.execute(query_seasonality, {"id_clean": id_real}).fetchall()
            exo_rows = conn.execute(query_exo).fetchall()

        base_val_unitario = float(prod_row[1] or 10000.0)
        sim_val_unitario = base_val_unitario * (1.0 + (precio_pct / 100.0))

        base_units = float(stat_row[0] if stat_row[0] is not None else 50.0)
        l1 = float(stat_row[1] if stat_row[1] is not None else base_units)
        l2 = float(stat_row[2] if stat_row[2] is not None else base_units)
        ma3 = float(stat_row[3] if stat_row[3] is not None else base_units)

        start_year = int(stat_row[4]) if stat_row[4] is not None else 2025
        start_month = int(stat_row[5]) if stat_row[5] is not None else 12

        seasonality_map = {row[0]: float(row[1]) for row in seas_rows}
        exo_map = {row[0]: row for row in exo_rows}

        cur_date = datetime.date(start_year, start_month, 1)
        simulados = []

        macro_elasticity_factor = max(
            0.40,
            min(
                1.80,
                1.0 
                - (0.50 * (precio_pct / 100.0))
                - (0.20 * (usd_pct / 100.0))
                - (0.25 * (ipc_pct / 100.0))
            )
        )

        for _ in range(1, 13):
            if cur_date.month == 12:
                cur_date = datetime.date(cur_date.year + 1, 1, 1)
            else:
                cur_date = datetime.date(cur_date.year, cur_date.month + 1, 1)

            p_anio = cur_date.year
            p_mes = cur_date.month

            l12 = seasonality_map.get(p_mes, base_units)

            if p_mes in exo_map:
                e = exo_map[p_mes]
                t_max = float(e[1])
                t_med = float(e[2])
                precip = float(e[3])
                ipc_ind = float(e[4]) * (1.0 + (ipc_pct / 100.0))
                ipc_var = float(e[5]) * (1.0 + (ipc_pct / 100.0))
                tc_usd = float(e[6]) * (1.0 + (usd_pct / 100.0))
                n_feriados = float(e[7])
                has_festivo = float(e[8])
            else:
                t_max, t_med, precip = 30.0, 25.0, 100.0
                ipc_ind = 100.0 * (1.0 + (ipc_pct / 100.0))
                ipc_var = 4.5 * (1.0 + (ipc_pct / 100.0))
                tc_usd = 7200.0 * (1.0 + (usd_pct / 100.0))
                n_feriados, has_festivo = 0.0, 0.0

            seasonal_ratio = (l12 / base_units) if base_units > 0 else 1.0
            seasonal_ratio = max(0.70, min(1.40, seasonal_ratio))

            feature_vector = np.array([[
                l1 * macro_elasticity_factor,
                l2 * macro_elasticity_factor,
                l12 * macro_elasticity_factor,
                ma3 * macro_elasticity_factor,
                t_max, t_med, precip,
                ipc_ind, ipc_var, tc_usd,
                n_feriados, has_festivo
            ]])

            if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                tree_preds_raw = np.array([tree.predict(feature_vector)[0] for tree in model.estimators_])
                tree_preds_units = np.expm1(np.clip(tree_preds_raw, 0, None)) * seasonal_ratio
                p10 = float(np.percentile(tree_preds_units, 10))
                p50 = float(np.percentile(tree_preds_units, 50))
                p90 = float(np.percentile(tree_preds_units, 90))
                pred_unidades = round(p50, 2) if p50 > 0 else max(1.0, round(l12 * macro_elasticity_factor, 2))
                sim_inf = round(max(0.0, p10), 2)
                sim_sup = round(max(pred_unidades, p90), 2)
            else:
                pred_raw_log = float(model.predict(feature_vector)[0])
                pred_raw = float(np.expm1(max(0.0, pred_raw_log)))
                if pred_raw <= 0:
                    pred_unidades = max(1.0, round(l12 * macro_elasticity_factor, 2))
                else:
                    pred_unidades = round(pred_raw * seasonal_ratio, 2)
                sim_inf = round(pred_unidades * 0.85, 2)
                sim_sup = round(pred_unidades * 1.15, 2)

            monto_simulado = round(pred_unidades * sim_val_unitario, 2)

            simulados.append({
                "anio": p_anio,
                "mes": p_mes,
                "unidades_simuladas": pred_unidades,
                "monto_simulado": monto_simulado,
                "precio_unitario_simulado": round(sim_val_unitario, 2),
                "intervalo_confianza_inf": sim_inf,
                "intervalo_confianza_sup": sim_sup,
            })

            l2 = l1
            l1 = pred_unidades
            ma3 = round((l1 + l2) / 2.0, 2)

        return simulados

    def get_model_explainability(self) -> Dict[str, Any]:
        global _CACHED_MODEL
        model = _CACHED_MODEL
        if model is None:
            model, _, _, _ = self.train_and_evaluate()

        feature_metadata = {
            "lag_1": {"nombre": "Ventas del Mes Anterior (t-1)", "grupo": "Historial y Tendencia"},
            "lag_2": {"nombre": "Ventas Hace 2 Meses (t-2)", "grupo": "Historial y Tendencia"},
            "lag_12": {"nombre": "Ventas Mismo Mes Año Anterior (t-12)", "grupo": "Historial y Tendencia"},
            "ma_3m": {"nombre": "Media Móvil Trimestral", "grupo": "Historial y Tendencia"},
            "temp_max_promedio": {"nombre": "Temperatura Máxima Promedio", "grupo": "Variables Climáticas"},
            "temp_media_promedio": {"nombre": "Temperatura Media Promedio", "grupo": "Variables Climáticas"},
            "precipitacion_mm": {"nombre": "Precipitación Acumulada (mm)", "grupo": "Variables Climáticas"},
            "ipc_indice": {"nombre": "Índice de Precios al Consumidor (IPC)", "grupo": "Variables Macroeconómicas"},
            "ipc_var_interanual": {"nombre": "Inflación Interanual (%)", "grupo": "Variables Macroeconómicas"},
            "tipo_cambio_pyg_usd": {"nombre": "Tipo de Cambio (PYG / USD)", "grupo": "Variables Macroeconómicas"},
            "cantidad_feriados": {"nombre": "Días Feriados en el Mes", "grupo": "Calendario y Festividades"},
            "tiene_festivo_comercial": {"nombre": "Festivo Comercial Clave", "grupo": "Calendario y Festividades"},
        }

        if hasattr(model, "feature_importances_"):
            raw_importances = model.feature_importances_
            total = np.sum(raw_importances)
            normalized = (raw_importances / total * 100.0) if total > 0 else raw_importances
        elif hasattr(model, "coef_"):
            raw_coefs = np.abs(model.coef_)
            total = np.sum(raw_coefs)
            normalized = (raw_coefs / total * 100.0) if total > 0 else raw_coefs
        else:
            normalized = np.ones(len(self.FEATURE_COLS)) / len(self.FEATURE_COLS) * 100.0

        variables_list = []
        grupos_dict = {}

        for feat, val in sorted(zip(self.FEATURE_COLS, normalized), key=lambda x: x[1], reverse=True):
            pct_val = round(float(val), 2)
            meta = feature_metadata.get(feat, {"nombre": feat, "grupo": "General"})
            grupo_name = meta["grupo"]

            impacto_label = "Muy Alto" if pct_val >= 40.0 else ("Alto" if pct_val >= 10.0 else ("Moderado" if pct_val >= 1.0 else "Bajo"))

            variables_list.append({
                "feature": feat,
                "nombre_descriptivo": meta["nombre"],
                "grupo": grupo_name,
                "importancia_pct": pct_val,
                "nivel_impacto": impacto_label,
            })

            grupos_dict[grupo_name] = round(grupos_dict.get(grupo_name, 0.0) + pct_val, 2)

        return {
            "status": "success",
            "modelo_tipo": type(model).__name__,
            "total_variables": len(self.FEATURE_COLS),
            "resumen_por_grupo": grupos_dict,
            "importancia_variables": variables_list
        }


if __name__ == "__main__":
    runner = DbTrainerPredictor()
    model, metrics, desde, hasta = runner.train_and_evaluate()
    company_model_map = runner.save_model_version("baseline_linear_v1", metrics, desde, hasta)
    runner.generate_and_save_predictions(model, company_model_map)
