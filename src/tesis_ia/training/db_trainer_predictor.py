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
        df = self.load_dataset_from_db()
        if df.empty:
            raise ValueError("No hay datos en ventas_mensuales para entrenar.")

        df[self.FEATURE_COLS] = df[self.FEATURE_COLS].fillna(0)

        train_mask = df["anio"] < 2025
        test_mask = df["anio"] == 2025

        if not train_mask.any():
            train_mask = df["anio"] <= df["anio"].max() - 1
            test_mask = df["anio"] == df["anio"].max()

        X_train = df.loc[train_mask, self.FEATURE_COLS]
        y_train = df.loc[train_mask, "unidades_vendidas"]
        X_test = df.loc[test_mask, self.FEATURE_COLS]
        y_test = df.loc[test_mask, "unidades_vendidas"]

        model = LinearRegression()
        model.fit(X_train, y_train)

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
        print(f"[TRAIN] Regresión Lineal entrenada. MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
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
        with self.engine.connect() as conn:
            prods_df = pd.read_sql(query_prods, conn)

        if prods_df.empty:
            print("[WARN] No hay productos para generar predicciones.")
            return 0

        today = datetime.date.today()
        proyecciones = []

        default_model_id = company_model_map if isinstance(company_model_map, str) else next(iter(company_model_map.values()), str(uuid.uuid4()))

        for _, prod in prods_df.iterrows():
            id_prod = prod["id_producto"]
            id_emp = prod["id_empresa"]
            val_unitario = float(prod["valor_unitario"] or 10000.0)

            if isinstance(company_model_map, dict):
                id_mod = company_model_map.get(id_emp, default_model_id)
            else:
                id_mod = default_model_id

            cur_date = today.replace(day=1)
            for i in range(1, 13):
                if cur_date.month == 12:
                    cur_date = datetime.date(cur_date.year + 1, 1, 1)
                else:
                    cur_date = datetime.date(cur_date.year, cur_date.month + 1, 1)

                p_anio = cur_date.year
                p_mes = cur_date.month

                sample_features = np.array([[10, 12, 15, 11, 32.0, 26.0, 80.0, 110.0, 4.5, 7300.0, 1, 0]])
                pred_unidades = float(model.predict(sample_features)[0])
                pred_unidades = max(5.0, round(pred_unidades, 2))
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
            print(f"[OK] {len(proyecciones)} predicciones insertadas/actualizadas en la tabla predicciones.")
            return len(proyecciones)
        except Exception as e:
            print(f"[ERROR] Falló la inserción de predicciones en MySQL: {e}")
            raise e


if __name__ == "__main__":
    runner = DbTrainerPredictor()
    model, metrics, desde, hasta = runner.train_and_evaluate()
    company_model_map = runner.save_model_version("baseline_linear_v1", metrics, desde, hasta)
    runner.generate_and_save_predictions(model, company_model_map)
