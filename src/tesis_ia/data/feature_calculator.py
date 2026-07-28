"""Módulo de cálculo y actualización de features (Lags, Media Móvil, Market Share) en ventas_mensuales."""

from __future__ import annotations

from typing import Dict, Any
from sqlalchemy import text
from tesis_ia.utils.db_connector import get_engine


class FeatureCalculator:

    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def consolidate_weekly_sales(self, anio: int, mes: int) -> int:
        sql = text("""
            INSERT INTO ventas_mensuales (
                id_producto, id_archivo, id_empresa, anio, mes,
                unidades_vendidas, monto_facturado, precio_unitario, es_agregado
            )
            SELECT
                vs.id_producto,
                vs.id_archivo,
                vs.id_empresa,
                vs.anio,
                :mes AS mes,
                SUM(vs.unidades_vendidas) AS unidades_vendidas,
                SUM(vs.monto_facturado) AS monto_facturado,
                CASE WHEN SUM(vs.unidades_vendidas) > 0 THEN SUM(vs.monto_facturado) / SUM(vs.unidades_vendidas) ELSE 0 END AS precio_unitario,
                1 AS es_agregado
            FROM ventas_semanales vs
            -- Aproximación de semanas pertenecientes al mes (ej: semana 1-4 mes 1, etc.)
            WHERE vs.anio = :anio AND FLOOR((vs.semana - 1) / 4.33) + 1 = :mes
            GROUP BY vs.id_producto, vs.id_archivo, vs.id_empresa, vs.anio
            ON DUPLICATE KEY UPDATE
                unidades_vendidas = VALUES(unidades_vendidas),
                monto_facturado = VALUES(monto_facturado),
                precio_unitario = VALUES(precio_unitario),
                es_agregado = 1;
        """)
        try:
            with self.engine.begin() as conn:
                res = conn.execute(sql, {"anio": anio, "mes": mes})
                return res.rowcount
        except Exception as e:
            print(f"[ERROR] Falló la consolidación semanal a mensual para {anio}-{mes}: {e}")
            return 0

    def compute_features_month(self, anio: int, mes: int) -> bool:
        sql_lags = text("""
            UPDATE ventas_mensuales vm
            LEFT JOIN ventas_mensuales l1 
                ON vm.id_producto = l1.id_producto AND vm.id_empresa = l1.id_empresa
                AND ((vm.mes = 1 AND l1.anio = vm.anio - 1 AND l1.mes = 12) OR (vm.mes > 1 AND l1.anio = vm.anio AND l1.mes = vm.mes - 1))
            LEFT JOIN ventas_mensuales l2
                ON vm.id_producto = l2.id_producto AND vm.id_empresa = l2.id_empresa
                AND ((vm.mes <= 2 AND l2.anio = vm.anio - 1 AND l2.mes = vm.mes + 10) OR (vm.mes > 2 AND l2.anio = vm.anio AND l2.mes = vm.mes - 2))
            LEFT JOIN ventas_mensuales l12
                ON vm.id_producto = l12.id_producto AND vm.id_empresa = l12.id_empresa
                AND l12.anio = vm.anio - 1 AND l12.mes = vm.mes
            SET 
                vm.lag_1 = l1.unidades_vendidas,
                vm.lag_2 = l2.unidades_vendidas,
                vm.lag_12 = l12.unidades_vendidas,
                vm.ma_3m = (COALESCE(l1.unidades_vendidas, 0) + COALESCE(l2.unidades_vendidas, 0)) / 2.0
            WHERE vm.anio = :anio AND vm.mes = :mes;
        """)

        sql_market_share = text("""
            UPDATE ventas_mensuales vm
            JOIN (
                SELECT 
                    pv.id_categoria,
                    v.anio,
                    v.mes,
                    SUM(v.unidades_vendidas) AS total_cat_unidades
                FROM ventas_mensuales v
                JOIN productos_versiones pv ON v.id_producto = pv.id_producto
                WHERE v.anio = :anio AND v.mes = :mes
                GROUP BY pv.id_categoria, v.anio, v.mes
            ) cat_totales ON cat_totales.anio = vm.anio AND cat_totales.mes = vm.mes
            JOIN productos_versiones pv2 ON vm.id_producto = pv2.id_producto AND pv2.id_categoria = cat_totales.id_categoria
            SET vm.market_share_unidades = CASE 
                WHEN cat_totales.total_cat_unidades > 0 
                THEN (vm.unidades_vendidas * 1.0) / cat_totales.total_cat_unidades 
                ELSE 0 
            END
            WHERE vm.anio = :anio AND vm.mes = :mes;
        """)

        sql_price_gap = text("""
            UPDATE ventas_mensuales vm
            JOIN (
                SELECT 
                    pv.id_categoria,
                    v.anio,
                    v.mes,
                    AVG(v.precio_unitario) AS precio_prom_cat
                FROM ventas_mensuales v
                JOIN productos_versiones pv ON v.id_producto = pv.id_producto
                WHERE v.anio = :anio AND v.mes = :mes AND v.precio_unitario IS NOT NULL
                GROUP BY pv.id_categoria, v.anio, v.mes
            ) cat_precios ON cat_precios.anio = vm.anio AND cat_precios.mes = vm.mes
            JOIN productos_versiones pv2 ON vm.id_producto = pv2.id_producto AND pv2.id_categoria = cat_precios.id_categoria
            SET vm.price_gap_pct = CASE 
                WHEN cat_precios.precio_prom_cat > 0 
                THEN (vm.precio_unitario - cat_precios.precio_prom_cat) / cat_precios.precio_prom_cat 
                ELSE 0 
            END
            WHERE vm.anio = :anio AND vm.mes = :mes;
        """)

        try:
            with self.engine.begin() as conn:
                conn.execute(sql_lags, {"anio": anio, "mes": mes})
                conn.execute(sql_market_share, {"anio": anio, "mes": mes})
                conn.execute(sql_price_gap, {"anio": anio, "mes": mes})
            print(f"[OK] Features (lags, ma_3m, market share, price gap) actualizados para {anio}-{mes:02d}.")
            return True
        except Exception as e:
            print(f"[ERROR] Falló el cálculo de features para {anio}-{mes}: {e}")
            return False


if __name__ == "__main__":
    calc = FeatureCalculator()
    calc.compute_features_month(2025, 1)
