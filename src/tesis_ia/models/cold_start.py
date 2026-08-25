"""Módulo de Cold Start para productos nuevos sin historial de ventas.

Utiliza K-Means para segmentación de catálogo y Nearest Neighbors (KNN)
sobre el espacio vectorial de Departamento, Categoría, Marca, Contenido Neto y Precio
para transferir la curva de demanda de productos 'gemelos' maduros hacia el nuevo lanzamiento.
"""

from __future__ import annotations

import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from tesis_ia.utils.db_connector import get_engine


class ColdStartEngine:

    def __init__(self, engine=None):
        self.engine = engine or get_engine()
        self.pipeline: Optional[ColumnTransformer] = None
        self.kmeans: Optional[KMeans] = None
        self.knn: Optional[NearestNeighbors] = None
        self.catalog_df: Optional[pd.DataFrame] = None
        self.seasonality_map: Dict[Tuple[str, int], float] = {}
        self.is_fitted = False

    def load_catalog_and_history(self) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], float]]:
        query_catalog = """
            SELECT 
                p.id AS id_producto,
                p.sku,
                p.descripcion,
                p.contenido_neto,
                p.unidad_medida,
                COALESCE(p.valor_unitario, 10000.0) AS valor_unitario,
                p.id_marca,
                COALESCE(m.nombre, 'Sin Marca') AS nombre_marca,
                m.id_empresa,
                COALESCE(pv.id_categoria, 1) AS id_categoria,
                COALESCE(c.nombre, 'General') AS nombre_categoria,
                COALESCE(c.id_departamento, 1) AS id_departamento,
                COALESCE(d.nombre, 'General') AS nombre_departamento,
                stats.avg_unidades,
                stats.total_unidades
            FROM productos p
            LEFT JOIN marcas m ON p.id_marca = m.id
            LEFT JOIN productos_versiones pv ON p.id = pv.id_producto
            LEFT JOIN categorias c ON pv.id_categoria = c.id
            LEFT JOIN departamentos d ON c.id_departamento = d.id
            INNER JOIN (
                SELECT 
                    id_producto,
                    AVG(unidades_vendidas) AS avg_unidades,
                    SUM(unidades_vendidas) AS total_unidades
                FROM ventas_mensuales
                WHERE unidades_vendidas > 0
                GROUP BY id_producto
            ) stats ON p.id = stats.id_producto
            WHERE p.contenido_neto > 0;
        """

        query_seasonality = """
            SELECT 
                id_producto,
                mes,
                AVG(unidades_vendidas) AS avg_mes
            FROM ventas_mensuales
            GROUP BY id_producto, mes;
        """

        with self.engine.connect() as conn:
            catalog_df = pd.read_sql(query_catalog, conn)
            seasonality_df = pd.read_sql(query_seasonality, conn)

        if catalog_df.empty:
            print("[WARN] Catálogo vacío o sin historial en MySQL para Cold Start.")
            return pd.DataFrame(), {}

        catalog_df["id_producto_clean"] = catalog_df["id_producto"].astype(str).str.strip().str.lower()
        seasonality_df["id_producto_clean"] = seasonality_df["id_producto"].astype(str).str.strip().str.lower()

        seasonality_map = seasonality_df.set_index(["id_producto_clean", "mes"])["avg_mes"].to_dict()

        return catalog_df, seasonality_map

    def fit(self) -> bool:
        catalog_df, seasonality_map = self.load_catalog_and_history()
        if catalog_df.empty:
            return False

        self.catalog_df = catalog_df
        self.seasonality_map = seasonality_map

        # Definición de features para similitud paramétrica
        numeric_features = ["log_precio", "log_contenido"]
        categorical_features = ["id_departamento", "id_categoria", "unidad_medida", "id_marca"]

        catalog_df["log_precio"] = np.log1p(catalog_df["valor_unitario"].clip(lower=100.0))
        catalog_df["log_contenido"] = np.log1p(catalog_df["contenido_neto"].clip(lower=1.0))

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )

        X_transformed = self.pipeline.fit_transform(catalog_df)

        n_samples = len(catalog_df)
        n_clusters = min(15, max(3, n_samples // 50))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.catalog_df["cluster_id"] = self.kmeans.fit_predict(X_transformed)

        self.knn = NearestNeighbors(n_neighbors=min(5, n_samples), metric="cosine")
        self.knn.fit(X_transformed)

        self.is_fitted = True
        print(f"[COLD-START] Motor ajustado con {n_samples:,} productos en {n_clusters} clústeres.")
        return True

    def predict_new_product(
        self,
        id_departamento: int,
        id_categoria: int,
        id_marca: str,
        contenido_neto: float,
        unidad_medida: str,
        valor_unitario: float,
        descripcion: str = "Nuevo Producto",
        horizonte_meses: int = 12,
        anio_inicio: Optional[int] = None,
        mes_inicio: Optional[int] = None,
        id_empresa: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.is_fitted:
            ok = self.fit()
            if not ok or self.catalog_df is None or self.knn is None or self.pipeline is None:
                raise RuntimeError("No se pudo inicializar el catálogo para el modelo de Cold Start.")

        now = datetime.datetime.now()
        start_year = anio_inicio or now.year
        start_month = mes_inicio or now.month

        new_prod_df = pd.DataFrame([{
            "id_departamento": id_departamento,
            "id_categoria": id_categoria,
            "unidad_medida": str(unidad_medida).lower(),
            "id_marca": str(id_marca),
            "log_precio": np.log1p(max(100.0, float(valor_unitario))),
            "log_contenido": np.log1p(max(1.0, float(contenido_neto))),
        }])

        X_new = self.pipeline.transform(new_prod_df)

        cluster_id = int(self.kmeans.predict(X_new)[0]) if self.kmeans else 0

        distances, indices = self.knn.kneighbors(X_new)
        peer_indices = indices[0]
        peer_distances = distances[0]

        similar_products = []
        peer_avg_units_list = []
        peer_skus_clean = []

        for idx, dist in zip(peer_indices, peer_distances):
            peer = self.catalog_df.iloc[idx]
            sim_pct = max(0.0, min(100.0, round((1.0 - float(dist)) * 100.0, 1)))
            
            similar_products.append({
                "id_producto": str(peer["id_producto"]),
                "sku": str(peer["sku"]),
                "descripcion": str(peer["descripcion"]),
                "departamento": str(peer["nombre_departamento"]),
                "categoria": str(peer["nombre_categoria"]),
                "marca": str(peer["nombre_marca"]),
                "contenido_neto": float(peer["contenido_neto"]),
                "unidad_medida": str(peer["unidad_medida"]),
                "valor_unitario": float(peer["valor_unitario"]),
                "ventas_promedio_mensual": float(peer["avg_unidades"]),
                "similitud_pct": sim_pct,
                "cluster_id": int(peer.get("cluster_id", cluster_id))
            })
            peer_avg_units_list.append(float(peer["avg_unidades"]))
            peer_skus_clean.append(str(peer["id_producto_clean"]))

        weights = [max(0.05, 1.0 - float(d)) for d in peer_distances]
        sum_weights = sum(weights) or 1.0
        normalized_weights = [w / sum_weights for w in weights]

        base_estimated_units = sum(u * w for u, w in zip(peer_avg_units_list, normalized_weights))

        avg_peers_price = np.mean([p["valor_unitario"] for p in similar_products]) or valor_unitario
        if avg_peers_price > 0:
            price_ratio = valor_unitario / avg_peers_price
            price_elasticity_factor = max(0.6, min(1.4, (1.0 / (price_ratio ** 0.4))))
        else:
            price_elasticity_factor = 1.0

        calibrated_base_units = max(1.0, round(base_estimated_units * price_elasticity_factor, 2))

        cur_date = datetime.date(start_year, start_month, 1)
        proyeccion_12_meses = []

        for _ in range(horizonte_meses):
            p_anio = cur_date.year
            p_mes = cur_date.month

            month_factors = []
            for sku_clean in peer_skus_clean:
                if (sku_clean, p_mes) in self.seasonality_map:
                    hist_month_avg = self.seasonality_map[(sku_clean, p_mes)]
                    par_base = self.catalog_df[self.catalog_df["id_producto_clean"] == sku_clean]["avg_unidades"].values
                    par_base_val = float(par_base[0]) if len(par_base) > 0 else calibrated_base_units
                    ratio = (hist_month_avg / par_base_val) if par_base_val > 0 else 1.0
                    month_factors.append(ratio)
                else:
                    month_factors.append(1.0)

            avg_month_factor = sum(f * w for f, w in zip(month_factors, normalized_weights))
            avg_month_factor = max(0.70, min(1.40, avg_month_factor))

            pred_unidades = round(calibrated_base_units * avg_month_factor, 2)
            monto_predicho = round(pred_unidades * valor_unitario, 2)

            proyeccion_12_meses.append({
                "anio": p_anio,
                "mes": p_mes,
                "unidades_predichas": pred_unidades,
                "monto_predicho": monto_predicho,
                "intervalo_confianza_inf": round(pred_unidades * 0.80, 2),
                "intervalo_confianza_sup": round(pred_unidades * 1.20, 2),
                "factor_estacional_aplicado": round(avg_month_factor, 3)
            })

            if cur_date.month == 12:
                cur_date = datetime.date(cur_date.year + 1, 1, 1)
            else:
                cur_date = datetime.date(cur_date.year, cur_date.month + 1, 1)

        return {
            "status": "success",
            "producto_nuevo": {
                "descripcion": descripcion,
                "id_departamento": id_departamento,
                "id_categoria": id_categoria,
                "id_marca": id_marca,
                "contenido_neto": contenido_neto,
                "unidad_medida": unidad_medida,
                "valor_unitario": valor_unitario,
                "cluster_asignado": cluster_id,
                "unidades_base_estimadas": calibrated_base_units,
                "factor_elasticidad_precio": round(price_elasticity_factor, 3)
            },
            "productos_similares_gemelos": similar_products,
            "proyeccion_12_meses": proyeccion_12_meses
        }


# Instancia Singleton en memoria para respuestas ultra rápidas (< 20ms)
_COLD_START_ENGINE: Optional[ColdStartEngine] = None


def get_cold_start_engine() -> ColdStartEngine:
    """Obtiene o inicializa el motor de Cold Start en memoria RAM."""
    global _COLD_START_ENGINE
    if _COLD_START_ENGINE is None:
        _COLD_START_ENGINE = ColdStartEngine()
    return _COLD_START_ENGINE
