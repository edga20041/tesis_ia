"""Módulo de entrenamiento de Regresión Lineal y generación de predicciones a 12 meses directamente en MySQL."""

from __future__ import annotations

import json
import uuid
import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text

from tesis_ia.utils.db_connector import get_engine

# Variable global en memoria para caché del modelo entrenado
_CACHED_MODEL: Optional[LinearRegression] = None


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

    def train_and_evaluate(self) -> Tuple[LinearRegression, Dict[str, float], str, str]:
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
        print(f"[TRAIN] Regresión Lineal entrenada (Ventana Móvil: Train < {cutoff_date.strftime('%Y-%m')}, Validation Test >= {cutoff_date.strftime('%Y-%m')}). MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return model, metrics, fecha_desde, fecha_hasta

    def save_model_version(self, version: str, metrics: Dict[str, float],
                           desde: str, hasta: str, id_empresa: Optional[str] = None) -> Dict[str, str]:
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
                activo = 1
        """)

        company_model_map: Dict[str, str] = {}

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
                    "hiperparametros": json.dumps({"family": "LinearRegression", "algorithm": "scikit-learn"})
                })

            rows = conn.execute(text("SELECT id_empresa, id FROM modelos_versiones WHERE activo = 1")).fetchall()
            for r_emp, r_id in rows:
                company_model_map[r_emp] = r_id

        print(f"[OK] Modelo '{version}' guardado como ACTIVO para {len(company_model_map)} empresa(s).")
        return company_model_map

    def generate_and_save_predictions(self, model: LinearRegression,
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

        proyecciones = []
        default_model_id = company_model_map if isinstance(company_model_map, str) else next(iter(company_model_map.values()), str(uuid.uuid4()))

        for _, prod in prods_merged.iterrows():
            id_prod = prod["id_producto"]
            id_prod_clean = prod["id_producto_clean"]
            id_emp = prod["id_empresa"]
            val_unitario = float(prod["valor_unitario"] or 10000.0)

            id_mod = company_model_map.get(id_emp, default_model_id) if isinstance(company_model_map, dict) else default_model_id

            base_units = float(prod["avg_unidades"] if pd.notnull(prod["avg_unidades"]) else 50.0)
            l1 = float(prod["last_lag_1"] if pd.notnull(prod["last_lag_1"]) else base_units)
            l2 = float(prod["last_lag_2"] if pd.notnull(prod["last_lag_2"]) else base_units)
            ma3 = float(prod["last_ma_3m"] if pd.notnull(prod["last_ma_3m"]) else base_units)
            
            start_year = int(prod["last_anio"]) if pd.notnull(prod["last_anio"]) else 2025
            start_month = int(prod["last_mes"]) if pd.notnull(prod["last_mes"]) else 12

            cur_date = datetime.date(start_year, start_month, 1)

            for i in range(1, 13):
                if cur_date.month == 12:
                    cur_date = datetime.date(cur_date.year + 1, 1, 1)
                else:
                    cur_date = datetime.date(cur_date.year, cur_date.month + 1, 1)

                p_anio = cur_date.year
                p_mes = cur_date.month

                if (id_prod_clean, p_mes) in seasonality_map:
                    l12 = float(seasonality_map[(id_prod_clean, p_mes)])
                else:
                    l12 = base_units

                if (p_anio, p_mes) in exo_exact_df.index:
                    exo_row = exo_exact_df.loc[(p_anio, p_mes)]
                    if isinstance(exo_row, pd.DataFrame):
                        exo_row = exo_row.iloc[0]
                    t_max = float(exo_row["temp_max_promedio"])
                    t_med = float(exo_row["temp_media_promedio"])
                    precip = float(exo_row["precipitacion_mm"])
                    ipc_ind = float(exo_row["ipc_indice"])
                    ipc_var = float(exo_row["ipc_var_interanual"])
                    tc_usd = float(exo_row["tipo_cambio_pyg_usd"])
                    n_feriados = float(exo_row["cantidad_feriados"])
                    has_festivo = float(exo_row["tiene_festivo_comercial"])

                elif p_mes in exo_month_df.index:
                    exo_row = exo_month_df.loc[p_mes]
                    if isinstance(exo_row, pd.DataFrame):
                        exo_row = exo_row.iloc[0]
                    t_max = float(exo_row["temp_max_promedio"])
                    t_med = float(exo_row["temp_media_promedio"])
                    precip = float(exo_row["precipitacion_mm"])
                    ipc_ind = float(exo_row["ipc_indice"])
                    ipc_var = float(exo_row["ipc_var_interanual"])
                    tc_usd = float(exo_row["tipo_cambio_pyg_usd"])
                    n_feriados = float(exo_row["cantidad_feriados"])
                    has_festivo = float(exo_row["tiene_festivo_comercial"])
                else:
                    t_max, t_med, precip = 30.0, 25.0, 100.0
                    ipc_ind, ipc_var, tc_usd = 100.0, 4.5, 7500.0
                    n_feriados, has_festivo = 0.0, 0.0

                feature_vector = np.array([[
                    l1, l2, l12, ma3,
                    t_max, t_med, precip,
                    ipc_ind, ipc_var, tc_usd,
                    n_feriados, has_festivo
                ]])

                pred_raw = float(model.predict(feature_vector)[0])

                if pred_raw <= 0:
                    pred_unidades = max(1.0, round(l12 * (1.0 + (has_festivo * 0.15) - (precip * 0.0005)), 2))
                else:
                    pred_unidades = round(pred_raw, 2)

                monto_predicho = round(pred_unidades * val_unitario, 2)
                inf = round(pred_unidades * 0.85, 2)
                sup = round(pred_unidades * 1.15, 2)

                pred_id = f"{id_prod}_{p_anio}_{p_mes}"

                proyecciones.append({
                    "id": pred_id,
                    "id_empresa": id_emp,
                    "id_producto": id_prod,
                    "id_modelo": id_mod,
                    "anio": p_anio,
                    "mes": p_mes,
                    "unidades_predichas": pred_unidades,
                    "monto_predicho": monto_predicho,
                    "intervalo_confianza_inf": inf,
                    "intervalo_confianza_sup": sup
                })

                l2 = l1
                l1 = pred_unidades
                ma3 = round((l1 + l2 + ma3) / 3.0, 2)

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
                batch_size = 5000
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
                pred_unidades = round(pred_raw, 2)

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
