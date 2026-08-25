"""Módulo de entrenamiento del Modelo Champion (Random Forest / Regresión Lineal) y generación de predicciones a 12 meses directamente en MySQL."""

from __future__ import annotations

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


class DbTrainerPredictor:

    FEATURE_COLS = [
        "lag_1", "lag_2", "lag_12", "ma_3m",
        "temp_max_promedio", "temp_media_promedio", "precipitacion_mm",
        "ipc_indice", "ipc_var_interanual", "tipo_cambio_pyg_usd",
        "cantidad_feriados", "tiene_festivo_comercial"
    ]

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

    def train_and_evaluate(self, model_family: str = "random_forest") -> Tuple[Any, Dict[str, float], str, str]:
        global _CACHED_MODEL
        df = self.load_dataset_from_db()
        if df.empty:
            raise ValueError("No hay datos en ventas_mensuales para entrenar.")

        df[self.FEATURE_COLS] = df[self.FEATURE_COLS].fillna(0)

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
            print("[TRAIN] Entrenando Champion: Random Forest Regressor (n_estimators=50, max_depth=10, n_jobs=1)...")
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                n_jobs=1,
                random_state=42
            )
        else:
            print("[TRAIN] Entrenando Baseline: Regresion Lineal Multivariada...")
            model = LinearRegression()

        model.fit(X_train, y_train)

        # Guardar en memoria RAM para uso inmediato en simulaciones
        _CACHED_MODEL = model

        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)  # No permitir ventas negativas

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
            hiperparams = json.dumps({"family": "RandomForestRegressor", "n_estimators": 100, "max_depth": 12, "algorithm": "scikit-learn"})
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
                id_producto,
                AVG(unidades_vendidas) AS avg_unidades,
                AVG(COALESCE(lag_1, unidades_vendidas)) AS last_lag_1,
                AVG(COALESCE(lag_2, unidades_vendidas)) AS last_lag_2,
                AVG(COALESCE(ma_3m, unidades_vendidas)) AS last_ma_3m,
                SUBSTRING_INDEX(MAX(CONCAT(anio, '-', LPAD(mes, 2, '0'))), '-', 1) AS last_anio,
                SUBSTRING_INDEX(MAX(CONCAT(anio, '-', LPAD(mes, 2, '0'))), '-', -1) AS last_mes
            FROM ventas_mensuales
            GROUP BY id_producto
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

            preds_raw = model.predict(X_step)

            # Modulación Estacional: preserva las oscilaciones orgánicas del mercado evitando que el árbol se estanque
            seasonal_ratio = np.clip(l12 / np.maximum(base_units, 1.0), 0.70, 1.40)
            pred_unidades = np.where(
                preds_raw > 0,
                np.round(preds_raw * seasonal_ratio, 2),
                np.maximum(1.0, np.round(l12 * (1.0 + has_festivo * 0.15 - precip * 0.0005), 2))
            )

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
                    "intervalo_confianza_inf": round(p_unidades * 0.85, 2),
                    "intervalo_confianza_sup": round(p_unidades * 1.15, 2),
                })

            l2 = l1.copy()
            l1 = pred_unidades.copy()
            ma3 = np.round((l1 + l2 + ma3) / 3.0, 2)

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

        query_stat = text("""
            SELECT 
                AVG(unidades_vendidas) AS avg_unidades,
                AVG(COALESCE(lag_1, unidades_vendidas)) AS last_lag_1,
                AVG(COALESCE(lag_2, unidades_vendidas)) AS last_lag_2,
                AVG(COALESCE(ma_3m, unidades_vendidas)) AS last_ma_3m,
                SUBSTRING_INDEX(MAX(CONCAT(anio, '-', LPAD(mes, 2, '0'))), '-', 1) AS last_anio,
                SUBSTRING_INDEX(MAX(CONCAT(anio, '-', LPAD(mes, 2, '0'))), '-', -1) AS last_mes
            FROM ventas_mensuales
            WHERE LOWER(id_producto) = :id_clean
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

        price_elasticity_factor = 1.0 - (0.5 * (precio_pct / 100.0))

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
                l1 * price_elasticity_factor,
                l2 * price_elasticity_factor,
                l12 * price_elasticity_factor,
                ma3 * price_elasticity_factor,
                t_max, t_med, precip,
                ipc_ind, ipc_var, tc_usd,
                n_feriados, has_festivo
            ]])

            pred_raw = float(model.predict(feature_vector)[0])
            if pred_raw <= 0:
                pred_unidades = max(1.0, round(l12 * price_elasticity_factor, 2))
            else:
                pred_unidades = round(pred_raw * seasonal_ratio, 2)

            monto_simulado = round(pred_unidades * sim_val_unitario, 2)

            simulados.append({
                "anio": p_anio,
                "mes": p_mes,
                "unidades_simuladas": pred_unidades,
                "monto_simulado": monto_simulado,
                "precio_unitario_simulado": round(sim_val_unitario, 2)
            })

            l2 = l1
            l1 = pred_unidades
            ma3 = round((l1 + l2 + ma3) / 3.0, 2)

        return simulados


if __name__ == "__main__":
    runner = DbTrainerPredictor()
    model, metrics, desde, hasta = runner.train_and_evaluate()
    company_model_map = runner.save_model_version("baseline_linear_v1", metrics, desde, hasta)
    runner.generate_and_save_predictions(model, company_model_map)
