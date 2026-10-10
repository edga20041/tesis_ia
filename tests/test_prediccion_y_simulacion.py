"""Pruebas del cierre mensual (generate_and_save_predictions) y del simulador
what-if (simulate_product_scenarios).

Son los dos consumidores reales del modelo Champion en producción: el primero
corre una vez al mes para todo el catálogo (cron_cierre_mensual.sh -> endpoint
FastAPI) y el segundo atiende cada ajuste de los sliders de precio/USD/IPC en
el dashboard. Ambos dependen de MySQL, asi que se simula el motor para
ejercitar su logica (pronostico a 12 meses, modulacion estacional, bandas de
confianza por cuantiles, elasticidades fijas) sin una base real.
"""

import datetime

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestRegressor

import tesis_ia.training.db_trainer_predictor as dtp
from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor


@pytest.fixture(autouse=True)
def limpiar_cache_de_modelo():
    dtp._CACHED_MODEL = None
    dtp._CACHED_RATIO_MODEL = None
    yield
    dtp._CACHED_MODEL = None
    dtp._CACHED_RATIO_MODEL = None


@pytest.fixture
def predictor():
    return DbTrainerPredictor(engine=MagicMock(name="engine"))


def modelo_bosque_entrenado(n_filas=60, n_estimadores=5):
    """Un RandomForest real y chico: hace falta ``estimators_`` para ejercitar
    la rama de cuantiles empíricos (P10/P50/P90), no alcanza con un mock."""
    rng = np.random.default_rng(0)
    X = rng.random((n_filas, 12))
    y = rng.random(n_filas) * 3.0  # target log-transformado
    modelo = RandomForestRegressor(n_estimators=n_estimadores, max_depth=3, random_state=0)
    modelo.fit(X, y)
    return modelo


def especialista_de_ratio_entrenado(n_filas=60, n_estimadores=5):
    """El especialista de ratio opera sobre FEATURE_COLS + EXTRA_FEATURE_COLS
    (18 columnas), no sobre las 12 de FEATURE_COLS."""
    rng = np.random.default_rng(1)
    X = rng.random((n_filas, 18))
    y = rng.random(n_filas) * 3.0
    modelo = RandomForestRegressor(n_estimators=n_estimadores, max_depth=3, random_state=1)
    modelo.fit(X, y)
    return modelo


class TestGenerateAndSavePredictions:
    def _mockear_lecturas(self, monkeypatch, prods_df, stats_df, seasonality_df, exo_exact_df, exo_month_df):
        lecturas = iter([prods_df, stats_df, seasonality_df, exo_exact_df, exo_month_df])
        monkeypatch.setattr(dtp.pd, "read_sql", MagicMock(side_effect=lambda *a, **k: next(lecturas)))

    def _dataframes_basicos(self, productos=("p1", "p2")):
        prods_df = pd.DataFrame({
            "id_producto": list(productos),
            "id_empresa": ["EMP-1"] * len(productos),
            "valor_unitario": [1000.0] * len(productos),
        })
        stats_df = pd.DataFrame({
            "id_producto": list(productos),
            "avg_unidades": [50.0] * len(productos),
            "last_lag_1": [55.0] * len(productos),
            "last_lag_2": [52.0] * len(productos),
            "last_ma_3m": [53.0] * len(productos),
            "last_market_share": [0.1] * len(productos),
            "last_price_gap": [0.0] * len(productos),
            "last_anio": [2025] * len(productos),
            "last_mes": [6] * len(productos),
        })
        seasonality_df = pd.DataFrame({
            "id_producto": list(productos) * 12,
            "mes": list(range(1, 13)) * len(productos),
            "avg_mes": [50.0] * (12 * len(productos)),
        })
        exo_exact_df = pd.DataFrame(columns=[
            "anio", "mes", "temp_max_promedio", "temp_media_promedio", "precipitacion_mm",
            "ipc_indice", "ipc_var_interanual", "tipo_cambio_pyg_usd",
            "cantidad_feriados", "tiene_festivo_comercial",
        ])
        exo_month_df = pd.DataFrame({
            "mes": list(range(1, 13)),
            "temp_max_promedio": [30.0] * 12,
            "temp_media_promedio": [25.0] * 12,
            "precipitacion_mm": [100.0] * 12,
            "ipc_indice": [100.0] * 12,
            "ipc_var_interanual": [4.5] * 12,
            "tipo_cambio_pyg_usd": [7200.0] * 12,
            "cantidad_feriados": [0] * 12,
            "tiene_festivo_comercial": [0] * 12,
        })
        return prods_df, stats_df, seasonality_df, exo_exact_df, exo_month_df

    def test_devuelve_cero_y_no_escribe_si_no_hay_productos(self, predictor, monkeypatch):
        prods_vacio = pd.DataFrame(columns=["id_producto", "id_empresa", "valor_unitario"])
        stats_vacio = pd.DataFrame(columns=["id_producto", "avg_unidades", "last_lag_1", "last_lag_2",
                                             "last_ma_3m", "last_market_share", "last_price_gap",
                                             "last_anio", "last_mes"])
        seasonality_vacio = pd.DataFrame(columns=["id_producto", "mes", "avg_mes"])
        _, _, _, exo_exact_vacio, exo_month_vacio = self._dataframes_basicos(productos=())
        self._mockear_lecturas(
            monkeypatch, prods_vacio, stats_vacio, seasonality_vacio, exo_exact_vacio, exo_month_vacio,
        )

        resultado = predictor.generate_and_save_predictions(model=MagicMock(), company_model_map={})

        assert resultado == 0
        predictor.engine.begin.assert_not_called()

    def test_inserta_doce_meses_por_producto_con_el_hibrido_de_bosque(self, predictor, monkeypatch):
        dfs = self._dataframes_basicos(productos=("p1", "p2"))
        self._mockear_lecturas(monkeypatch, *dfs)
        dtp._CACHED_RATIO_MODEL = especialista_de_ratio_entrenado()

        conn = MagicMock()
        predictor.engine.begin.return_value.__enter__.return_value = conn

        modelo = modelo_bosque_entrenado()
        total = predictor.generate_and_save_predictions(model=modelo, company_model_map="model-x")

        assert total == 24  # 2 productos x 12 meses
        assert conn.execute.call_count >= 2
        filas_insertadas = conn.execute.call_args_list[-1].args[1]
        assert len(filas_insertadas) == 24
        fila = filas_insertadas[0]
        assert fila["id_modelo"] == "model-x"
        assert fila["unidades_predichas"] >= 0
        assert fila["intervalo_confianza_inf"] <= fila["intervalo_confianza_sup"]
        assert dtp._CACHED_RATIO_MODEL is None

    def test_usa_el_mapa_de_empresa_a_modelo_cuando_es_un_diccionario(self, predictor, monkeypatch):
        prods_df = pd.DataFrame({
            "id_producto": ["p1", "p2"],
            "id_empresa": ["EMP-1", "EMP-2"],
            "valor_unitario": [1000.0, 2000.0],
        })
        _, stats_df, seasonality_df, exo_exact_df, exo_month_df = self._dataframes_basicos(("p1", "p2"))
        self._mockear_lecturas(monkeypatch, prods_df, stats_df, seasonality_df, exo_exact_df, exo_month_df)

        conn = MagicMock()
        predictor.engine.begin.return_value.__enter__.return_value = conn

        modelo = modelo_bosque_entrenado()
        predictor.generate_and_save_predictions(
            model=modelo, company_model_map={"EMP-1": "model-a", "EMP-2": "model-b"},
        )

        filas = conn.execute.call_args_list[-1].args[1]
        ids_por_empresa = {f["id_empresa"]: f["id_modelo"] for f in filas}
        assert ids_por_empresa == {"EMP-1": "model-a", "EMP-2": "model-b"}

    def test_usa_la_rama_de_respaldo_sin_ensamble_de_arboles(self, predictor, monkeypatch):
        # Un modelo sin `estimators_` (p.ej. regresion lineal) toma el camino
        # simple: banda de confianza +-15% en vez de percentiles por arbol.
        dfs = self._dataframes_basicos(productos=("p1",))
        self._mockear_lecturas(monkeypatch, *dfs)

        modelo_lineal = MagicMock(spec=["predict"])
        modelo_lineal.predict.return_value = np.array([3.0])  # log1p(unidades) ~ 3.0

        conn = MagicMock()
        predictor.engine.begin.return_value.__enter__.return_value = conn

        total = predictor.generate_and_save_predictions(model=modelo_lineal, company_model_map="model-x")

        assert total == 12
        filas = conn.execute.call_args_list[-1].args[1]
        for fila in filas:
            assert fila["intervalo_confianza_inf"] <= fila["unidades_predichas"] <= fila["intervalo_confianza_sup"]

    def test_relanza_la_excepcion_si_falla_la_insercion_y_libera_el_especialista(self, predictor, monkeypatch):
        dfs = self._dataframes_basicos(productos=("p1",))
        self._mockear_lecturas(monkeypatch, *dfs)
        dtp._CACHED_RATIO_MODEL = especialista_de_ratio_entrenado()

        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("boom")
        predictor.engine.begin.return_value.__enter__.return_value = conn

        with pytest.raises(RuntimeError, match="boom"):
            predictor.generate_and_save_predictions(model=modelo_bosque_entrenado(), company_model_map="model-x")

        assert dtp._CACHED_RATIO_MODEL is None


class TestSimulateProductScenarios:
    def _conn_falsa(self, prod_row, stat_row, seas_rows, exo_rows):
        conn = MagicMock()
        resultados = iter([prod_row, stat_row])  # ambos via fetchone()

        def execute(stmt, params=None):
            resultado = MagicMock()
            texto = str(stmt)
            if "FROM productos p" in texto:
                resultado.fetchone.return_value = prod_row
            elif "FROM ventas_mensuales" in texto and "GROUP BY mes" in texto:
                resultado.fetchall.return_value = seas_rows
            elif "FROM ventas_mensuales" in texto:
                resultado.fetchone.return_value = stat_row
            elif "variables_climaticas" in texto:
                resultado.fetchall.return_value = exo_rows
            return resultado

        conn.execute.side_effect = execute
        return conn

    def test_rechaza_un_producto_que_no_existe(self, predictor):
        conn = self._conn_falsa(prod_row=None, stat_row=None, seas_rows=[], exo_rows=[])
        predictor.engine.connect.return_value.__enter__.return_value = conn
        dtp._CACHED_MODEL = modelo_bosque_entrenado()

        with pytest.raises(ValueError, match="No se encontró"):
            predictor.simulate_product_scenarios("sku-inexistente")

    def test_devuelve_doce_meses_con_la_forma_esperada(self, predictor):
        prod_row = ("uuid-1", 1000.0, "EMP-1")
        stat_row = (50.0, 55.0, 52.0, 53.0, 2025, 6)
        seas_rows = [(m, 50.0) for m in range(1, 13)]
        exo_rows = [(m, 30.0, 25.0, 100.0, 100.0, 4.5, 7200.0, 0, 0) for m in range(1, 13)]
        conn = self._conn_falsa(prod_row, stat_row, seas_rows, exo_rows)
        predictor.engine.connect.return_value.__enter__.return_value = conn
        dtp._CACHED_MODEL = modelo_bosque_entrenado()

        resultado = predictor.simulate_product_scenarios("sku-1")

        assert len(resultado) == 12
        assert resultado[0]["anio"] == 2025 and resultado[0]["mes"] == 7
        claves = {"anio", "mes", "unidades_simuladas", "monto_simulado",
                  "precio_unitario_simulado", "intervalo_confianza_inf", "intervalo_confianza_sup"}
        assert set(resultado[0]) == claves

    def test_un_aumento_de_precio_reduce_la_demanda_vs_el_escenario_base(self, predictor):
        # ELASTICIDAD_PRECIO = -1.8: a mas precio, menos unidades, de forma
        # determinista independientemente de lo que prediga el bosque.
        prod_row = ("uuid-1", 1000.0, "EMP-1")
        stat_row = (50.0, 55.0, 52.0, 53.0, 2025, 6)
        seas_rows = [(m, 50.0) for m in range(1, 13)]
        exo_rows = [(m, 30.0, 25.0, 100.0, 100.0, 4.5, 7200.0, 0, 0) for m in range(1, 13)]

        modelo = modelo_bosque_entrenado()
        dtp._CACHED_MODEL = modelo

        conn_base = self._conn_falsa(prod_row, stat_row, seas_rows, exo_rows)
        predictor.engine.connect.return_value.__enter__.return_value = conn_base
        base = predictor.simulate_product_scenarios("sku-1", precio_pct=0.0)

        conn_subida = self._conn_falsa(prod_row, stat_row, seas_rows, exo_rows)
        predictor.engine.connect.return_value.__enter__.return_value = conn_subida
        subida = predictor.simulate_product_scenarios("sku-1", precio_pct=20.0)

        assert subida[0]["unidades_simuladas"] < base[0]["unidades_simuladas"]
        factor_esperado = 1.0 + DbTrainerPredictor.ELASTICIDAD_PRECIO * 0.20
        razon = subida[0]["unidades_simuladas"] / base[0]["unidades_simuladas"]
        assert razon == pytest.approx(factor_esperado, rel=1e-2)

    def test_el_precio_unitario_simulado_escala_con_precio_pct(self, predictor):
        prod_row = ("uuid-1", 1000.0, "EMP-1")
        stat_row = (50.0, 55.0, 52.0, 53.0, 2025, 6)
        seas_rows = [(m, 50.0) for m in range(1, 13)]
        exo_rows = [(m, 30.0, 25.0, 100.0, 100.0, 4.5, 7200.0, 0, 0) for m in range(1, 13)]
        conn = self._conn_falsa(prod_row, stat_row, seas_rows, exo_rows)
        predictor.engine.connect.return_value.__enter__.return_value = conn
        dtp._CACHED_MODEL = modelo_bosque_entrenado()

        resultado = predictor.simulate_product_scenarios("sku-1", precio_pct=10.0)

        assert resultado[0]["precio_unitario_simulado"] == pytest.approx(1100.0)

    def test_acepta_tanto_el_id_interno_como_el_sku(self, predictor):
        prod_row = ("uuid-1", 1000.0, "EMP-1")
        stat_row = (50.0, 55.0, 52.0, 53.0, 2025, 6)
        seas_rows = [(m, 50.0) for m in range(1, 13)]
        exo_rows = [(m, 30.0, 25.0, 100.0, 100.0, 4.5, 7200.0, 0, 0) for m in range(1, 13)]
        conn = self._conn_falsa(prod_row, stat_row, seas_rows, exo_rows)
        predictor.engine.connect.return_value.__enter__.return_value = conn
        dtp._CACHED_MODEL = modelo_bosque_entrenado()

        resultado = predictor.simulate_product_scenarios("  SKU-Con-Mayusculas  ")

        assert len(resultado) == 12

    def test_reentrena_si_no_hay_modelo_cacheado(self, predictor, monkeypatch):
        prod_row = ("uuid-1", 1000.0, "EMP-1")
        stat_row = (50.0, 55.0, 52.0, 53.0, 2025, 6)
        seas_rows = [(m, 50.0) for m in range(1, 13)]
        exo_rows = [(m, 30.0, 25.0, 100.0, 100.0, 4.5, 7200.0, 0, 0) for m in range(1, 13)]
        conn = self._conn_falsa(prod_row, stat_row, seas_rows, exo_rows)
        predictor.engine.connect.return_value.__enter__.return_value = conn

        modelo_de_respaldo = modelo_bosque_entrenado()
        monkeypatch.setattr(
            predictor, "train_and_evaluate",
            lambda *a, **k: (modelo_de_respaldo, {}, "2024-01-01", "2024-12-31"),
        )
        assert dtp._CACHED_MODEL is None

        resultado = predictor.simulate_product_scenarios("sku-1")

        assert len(resultado) == 12
