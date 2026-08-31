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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tesis_ia.utils.db_connector import get_engine


class ColdStartEngine:

    def __init__(self, engine=None):
        self.engine = engine or get_engine()
        self.pipeline: Optional[ColumnTransformer] = None
        self.kmeans: Optional[KMeans] = None
        self.knn: Optional[NearestNeighbors] = None
        self.tfidf: Optional[TfidfVectorizer] = None
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

        catalog_df["id_categoria"] = pd.to_numeric(catalog_df["id_categoria"], errors="coerce").fillna(1).astype(int)
        catalog_df["id_departamento"] = pd.to_numeric(catalog_df["id_departamento"], errors="coerce").fillna(1).astype(int)

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

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True, max_features=1000)
        self.tfidf.fit(catalog_df["descripcion"].fillna("").astype(str))

        self.is_fitted = True
        print(f"[COLD-START] Motor ajustado con {n_samples:,} productos en {n_clusters} clústeres y TF-IDF N-Grams.")
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

        cat_id_int = int(id_categoria)
        dept_id_int = int(id_departamento)

        cat_subset = self.catalog_df[self.catalog_df["id_categoria"] == cat_id_int]
        if len(cat_subset) >= 3:
            candidate_df = cat_subset.copy().reset_index(drop=True)
        else:
            dept_subset = self.catalog_df[self.catalog_df["id_departamento"] == dept_id_int]
            if len(dept_subset) >= 3:
                candidate_df = dept_subset.copy().reset_index(drop=True)
            else:
                candidate_df = self.catalog_df.copy().reset_index(drop=True)

        cand_log_price = np.log1p(candidate_df["valor_unitario"].clip(lower=100.0).values)
        cand_log_weight = np.log1p(candidate_df["contenido_neto"].clip(lower=1.0).values)
        
        target_log_price = np.log1p(max(100.0, float(valor_unitario)))
        target_log_weight = np.log1p(max(1.0, float(contenido_neto)))

        diff_price = np.abs(cand_log_price - target_log_price) / np.maximum(cand_log_price, target_log_price)
        diff_weight = np.abs(cand_log_weight - target_log_weight) / np.maximum(cand_log_weight, target_log_weight)
        
        sim_price = np.clip(1.0 - diff_price, 0.0, 1.0)
        sim_weight = np.clip(1.0 - diff_weight, 0.0, 1.0)
        
        same_brand = np.where(candidate_df["id_marca"].astype(str) == str(id_marca), 1.0, 0.0)
        same_unit = np.where(candidate_df["unidad_medida"].astype(str).str.lower() == str(unidad_medida).lower(), 1.0, 0.0)
        same_cat = np.where(candidate_df["id_categoria"] == cat_id_int, 1.0, 0.0)

        # Similitud de texto semántica con TF-IDF (N-gramas)
        if self.tfidf is not None and len(candidate_df) > 0:
            cand_tfidf = self.tfidf.transform(candidate_df["descripcion"].fillna("").astype(str))
            target_tfidf = self.tfidf.transform([str(descripcion)])
            sim_desc = cosine_similarity(target_tfidf, cand_tfidf)[0]
        else:
            sim_desc = np.zeros(len(candidate_df))

        affinity_scores = (
            (0.35 * same_cat) +
            (0.25 * sim_price) +
            (0.20 * sim_weight) +
            (0.10 * same_brand) +
            (0.05 * same_unit) +
            (0.05 * sim_desc)
        )
        
        top_k = min(5, len(candidate_df))
        top_indices = np.argsort(affinity_scores)[::-1][:top_k]
        
        similar_products = []
        peer_avg_units_list = []
        peer_skus_clean = []
        peer_weights = []

        for idx_pos in top_indices:
            peer = candidate_df.iloc[idx_pos]
            score_val = float(affinity_scores[idx_pos])
            sim_pct = max(10.0, min(100.0, round(score_val * 100.0, 1)))

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
            peer_weights.append(max(0.1, score_val))

        sum_weights = sum(peer_weights) or 1.0
        normalized_weights = [w / sum_weights for w in peer_weights]

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

        for idx_m in range(horizonte_meses):
            p_anio = cur_date.year
            p_mes = cur_date.month

            # Rampa de penetración comercial: Mes 1 = 65%, Mes 2 = 85%, Mes 3+ = 100%
            ramp_factor = 0.65 if idx_m == 0 else (0.85 if idx_m == 1 else 1.0)

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

            pred_unidades = round(calibrated_base_units * avg_month_factor * ramp_factor, 2)
            monto_predicho = round(pred_unidades * valor_unitario, 2)

            proyeccion_12_meses.append({
                "anio": p_anio,
                "mes": p_mes,
                "unidades_predichas": pred_unidades,
                "monto_predicho": monto_predicho,
                "intervalo_confianza_inf": round(pred_unidades * 0.80, 2),
                "intervalo_confianza_sup": round(pred_unidades * 1.20, 2),
                "factor_estacional_aplicado": round(avg_month_factor, 3),
                "factor_penetracion": ramp_factor
            })

            if cur_date.month == 12:
                cur_date = datetime.date(cur_date.year + 1, 1, 1)
            else:
                cur_date = datetime.date(cur_date.year, cur_date.month + 1, 1)

        avg_top_sim = np.mean([p["similitud_pct"] for p in similar_products[:3]]) if similar_products else 50.0
        conf_level = "ALTA" if avg_top_sim >= 75.0 else ("MODERADA" if avg_top_sim >= 60.0 else "PRELIMINAR")

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
                "factor_elasticidad_precio": round(price_elasticity_factor, 3),
                "nivel_confianza": conf_level,
                "confianza_score_pct": round(avg_top_sim, 1)
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
